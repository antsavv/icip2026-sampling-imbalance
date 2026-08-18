from os.path import join
import numpy as np
import time, pickle

from torch.utils.data import DataLoader, Dataset
import torch

from helper_tool import DataProcessing as DP
from helper_tool import ConfigDALES as cfg
from helper_ply import read_ply


# Read the subsampled data and divide the data into training and validation
class DALES(Dataset):
    def __init__(self, mode='training'):
        """
        DALES dataset for point cloud semantic segmentation.
        
        Args:
            mode: 'training', 'validation', or 'test'
        """
        self.name = 'DALES'
        self.path = 'data/DALES/'
        self.label_to_names = {0: 'Unknown',
                               1: 'Ground',
                               2: 'Vegetation',
                               3: 'Cars',
                               4: 'Trucks',
                               5: 'Power lines',
                               6: 'Fences',
                               7: 'Poles',
                               8: 'Buildings'}
        
        self.num_classes = len(self.label_to_names) - 1  # Exclude class 0 (Unknown)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])  # Ignore Unknown class

        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        weights = getattr(cfg, 'weights', 'none')
        cfg.class_weights = DP.get_class_weights('DALES', weights)
        cfg.name = 'DALES'

        # Fixed validation split (same as KPConv)
        self.validation_split = [0, 9, 13, 18, 22, 25]
        
        # Read file lists
        ply_path = join(self.path, 'dales_ply')
        with open(join(ply_path, 'train.txt'), 'r') as f:
            self.train_file_names = [line.strip() for line in f.readlines()]
        with open(join(ply_path, 'test.txt'), 'r') as f:
            self.test_file_names = [line.strip() for line in f.readlines()]
        
        # Determine which files to use based on mode
        self.mode = mode
        if mode == 'training':
            # Use training files excluding validation split
            self.all_files = [join(self.path, 'train_val_input_{:.3f}_randla-net'.format(cfg.sub_grid_size), 
                                  f + '.ply') for i, f in enumerate(self.train_file_names) 
                             if i not in self.validation_split]
            self.cloud_names = [f for i, f in enumerate(self.train_file_names) 
                               if i not in self.validation_split]
        elif mode == 'validation':
            # Use only validation split files
            self.all_files = [join(self.path, 'train_val_input_{:.3f}_randla-net'.format(cfg.sub_grid_size),
                                  f + '.ply') for i, f in enumerate(self.train_file_names) 
                             if i in self.validation_split]
            self.cloud_names = [f for i, f in enumerate(self.train_file_names) 
                               if i in self.validation_split]
        elif mode == 'test':
            # Use test files
            self.all_files = [join(self.path, 'test_input_{:.3f}_randla-net'.format(cfg.sub_grid_size),
                                  f + '.ply') for f in self.test_file_names]
            self.cloud_names = self.test_file_names
        else:
            raise ValueError(f'Unknown mode: {mode}. Choose from: training, validation, test')

        self.size = len(self.all_files)

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}
        
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

        print(f'Size of {mode} set: {len(self.input_labels[mode])}')
        
    def load_sub_sampled_clouds(self, sub_grid_size):
        """Load preprocessed point clouds and KDTree structures."""
        
        if self.mode in ['training', 'validation']:
            tree_path = join(self.path, 'train_val_input_{:.3f}_randla-net'.format(sub_grid_size))
        else:
            tree_path = join(self.path, 'test_input_{:.3f}_randla-net'.format(sub_grid_size))
        
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = self.cloud_names[i]

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Read PLY file
            data = read_ply(sub_ply_file)
            sub_labels = data['class']

            # Read KDTree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[self.mode] += [search_tree]
            self.input_labels[self.mode] += [sub_labels]
            self.input_names[self.mode] += [cloud_name]

            size = sub_labels.shape[0] * 4 * 4  # Approximate memory size
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(
                kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        # Load reprojection indices for validation and test
        if self.mode in ['validation', 'test']:
            print('\nPreparing reprojected indices for validation/testing')
            
            for i, cloud_name in enumerate(self.cloud_names):
                t0 = time.time()
                
                # Load projection file
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return self.size


class DALESSampler(Dataset):
    def __init__(self, dataset, split='training'):
        self.dataset = dataset
        self.split = split
        self.possibility = {}
        self.min_possibility = {}

        if split == 'training':
            self.num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split in ['validation', 'test']:
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size
        else:
            raise ValueError(f'Unknown split: {split}')

        self.possibility[split] = []
        self.min_possibility[split] = []
        
        # Initialize possibility for each point
        for i, tree in enumerate(self.dataset.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item, self.split)
        return selected_pc, selected_labels, selected_idx, cloud_ind

    def __len__(self):
        return self.num_per_epoch

    def spatially_regular_gen(self, item, split):
        """Generate spatially regular point samples."""
        
        # Choose the cloud with minimum possibility
        cloud_idx = int(np.argmin(self.min_possibility[split]))

        # Choose the point with minimum possibility as center point
        point_ind = np.argmin(self.possibility[split][cloud_idx])

        # Get all points from tree structure
        points = np.array(self.dataset.input_trees[split][cloud_idx].data, copy=False)

        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)

        # Query points
        if len(points) < cfg.num_points:
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

        # Shuffle indices
        queried_idx = DP.shuffle_idx(queried_idx)
        
        # Get corresponding points and labels
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point  # Center the points
        queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]

        # Update possibility
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[split][cloud_idx][queried_idx] += delta
        self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

        # Up-sample if necessary
        if len(points) < cfg.num_points:
            # For DALES, we don't have colors, so we pass None
            queried_pc_xyz, _, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, None, queried_pc_labels, queried_idx, cfg.num_points)

        # Convert to tensors
        queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
        queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
        queried_idx = torch.from_numpy(queried_idx).long()
        cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

        # For DALES: no color, use XYZ coordinates as features
        # You can also use height (Z) or other geometric features
        points = queried_pc_xyz  # Shape: (num_points, 3)

        return points, queried_pc_labels, queried_idx, cloud_idx

    def tf_map(self, batch_xyz, batch_label, batch_pc_idx, batch_cloud_idx):
        """Prepare network inputs with downsampling and KNN."""
        
        # For DALES, features are just XYZ coordinates
        batch_features = batch_xyz
        
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_xyz, 1)
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):
        """Collate batch data."""
        
        selected_pc, selected_labels, selected_idx, cloud_ind = [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack([pc.numpy() for pc in selected_pc])
        selected_labels = np.stack([label.numpy() for label in selected_labels])
        selected_idx = np.stack([idx.numpy() for idx in selected_idx])
        cloud_ind = np.stack([ci.numpy() for ci in cloud_ind])

        selected_xyz = selected_pc  # For DALES, points are just XYZ

        flat_inputs = self.tf_map(selected_xyz, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())

        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs


if __name__ == '__main__':
    # Test the dataset
    dataset = DALES('training')
    dataset_train = DALESSampler(dataset, split='training')
    dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, 
                           drop_last=True, collate_fn=dataset_train.collate_fn)
    
    for data in dataloader:
        features = data['features']
        labels = data['labels']
        idx = data['input_inds']
        cloud_idx = data['cloud_inds']
        print(f'Features shape: {features.shape}')
        print(f'Labels shape: {labels.shape}')
        print(f'Input indices shape: {idx.shape}')
        print(f'Cloud indices shape: {cloud_idx.shape}')
        break
