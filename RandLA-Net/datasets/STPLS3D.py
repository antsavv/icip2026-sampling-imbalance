from os.path import join
import numpy as np
import time
import pickle

from torch.utils.data import DataLoader, Dataset
import torch

from helper_tool import DataProcessing as DP
from helper_tool import ConfigSTPLS3D as cfg
from helper_ply import read_ply


class STPLS3D(Dataset):
    def __init__(self, mode='training'):

        self.name = 'STPLS3D'
        self.path = 'data/STPLS3D/'
        self.label_to_names = {0: 'ground',
                               1: 'building',
                               2: 'vegetation',
                               3: 'cars',
                               4: 'lightStreetSigns',
                               5: 'fences'}
        
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])  # No ignored labels in STPLS3D
        
        # Set config values
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        weights = getattr(cfg, 'weights', 'none')
        cfg.class_weights = DP.get_class_weights('STPLS3D', weights)
        cfg.name = 'STPLS3D'
        
        # Read file lists
        with open(join(self.path, 'train.txt'), 'r') as f:
            self.train_file_names = [line.strip() for line in f.readlines() if line.strip()]
        with open(join(self.path, 'test.txt'), 'r') as f:
            self.test_file_names = [line.strip() for line in f.readlines() if line.strip()]
        
        # Validation split - same as test (following KPConv pattern)
        self.val_split = self.test_file_names
        
        # Determine which files to use based on mode
        self.mode = mode
        tree_path = join(self.path, 'input_{:.3f}_randla-net'.format(cfg.sub_grid_size))
        
        if mode == 'training':
            # Use training files excluding validation split
            self.all_files = [join(tree_path, f + '.ply') 
                             for f in self.train_file_names 
                             if f not in self.val_split]
            self.cloud_names = [f for f in self.train_file_names 
                               if f not in self.val_split]
            
        elif mode == 'validation':
            # Use validation split files
            self.all_files = [join(tree_path, f + '.ply') 
                             for f in self.val_split]
            self.cloud_names = self.val_split.copy()

        elif mode == 'test':
            # Use test files (same as validation for STPLS3D)
            self.all_files = [join(tree_path, f + '.ply') 
                             for f in self.test_file_names]
            self.cloud_names = self.test_file_names.copy()

        else:
            raise ValueError(f'Unknown mode: {mode}. Choose from: training, validation, test')
        
        self.size = len(self.all_files)
        
        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}
        
        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        
        print(f'Size of {mode} set: {len(self.input_labels[mode])}')
    
    def load_sub_sampled_clouds(self, sub_grid_size):
        
        tree_path = join(self.path, 'input_{:.3f}_randla-net'.format(sub_grid_size))
        
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = self.cloud_names[i]
            
            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))
            
            # Read PLY file
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']
            
            # Read KDTree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)
            
            self.input_trees[self.mode] += [search_tree]
            self.input_colors[self.mode] += [sub_colors]
            self.input_labels[self.mode] += [sub_labels]
            self.input_names[self.mode] += [cloud_name]
            
            size = sub_colors.shape[0] * 4 * 7  # Approximate memory size
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))
        
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


class STPLS3DSampler(Dataset):
    """
    Sampler for STPLS3D dataset. Implements spatially regular sampling strategy following the RandLA-Net approach.
    """
    
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
        
        # Initialize possibility for each point
        self.possibility[split] = []
        self.min_possibility[split] = []
        
        for i, tree in enumerate(self.dataset.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]
    
    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item, self.split)
        return selected_pc, selected_labels, selected_idx, cloud_ind
    
    def __len__(self):
        return self.num_per_epoch
    
    def spatially_regular_gen(self, item, split):

        # Choose a random cloud         # Select scene containing the point with minimum possibility
        cloud_idx = int(np.argmin(self.min_possibility[split]))     

        # choose the point with the minimum of possibility in the cloud as query point  Select the minimum probability point in that scene as query point, point_ind is the point index
        point_ind = np.argmin(self.possibility[split][cloud_idx])

        # Get all points within the cloud from tree structure   Get xyz coordinates of all points in this scene from kdtree
        points = np.array(self.dataset.input_trees[split][cloud_idx].data, copy=False)

        # Center point of input region  Select the lowest probability point from all points (using index computed above), center_point shape is (1,3)
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)                    # Add noise

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < cfg.num_points:    # Take at most 40960 points (not all scenes have 40960 points, take all if not enough)
            # Query all points within the cloud
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

        # Shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)       # Shuffle and reassign the indices
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]            # Shuffle xyz info, use list as index where each number indexes matrix rows (first axis), returns in order, used to shuffle matrix
        queried_pc_xyz = queried_pc_xyz - pick_point    # Subtract center point, decentralization
        queried_pc_colors = self.dataset.input_colors[split][cloud_idx][queried_idx]
        queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)    # Calculate distance of each point from center point
        delta = np.square(1 - dists / np.max(dists))    # Note order of operations. Cleverly calculate probability update magnitude (farther from center = smaller probability increase = easier to be selected as center next time)
        self.possibility[split][cloud_idx][queried_idx] += delta    # Update probability so next center point selection doesn't repeat
        self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))  # Update minimum probability of this scene

        # up_sampled with replacement
        if len(points) < cfg.num_points:    # If less than 40960 points, use data augmentation to reach this count
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points) 

        queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()           # Convert back to tensor format
        queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
        queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
        queried_idx = torch.from_numpy(queried_idx).long()
        cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

        points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)
    
        return points, queried_pc_labels, queried_idx, cloud_idx
    
    def tf_map(self, batch_xyz, batch_features, batch_label, batch_pc_idx, batch_cloud_idx):
        """Prepare network inputs with downsampling and KNN."""
        
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):     # Downsampling at each layer implemented here (from here on, matrix order cannot be arbitrarily shuffled since KNN search relies on matrix indices to find neighboring points)
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)      # KNN search for 16 points around each point, record point indices, dimension is (6, 40960, 16)
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # Random downsampling, dimension is (6, 40960//4, 3)
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # Random downsampling on indices too (6, 40960//4, 16)
            up_i = DP.knn_search(sub_points, batch_xyz, 1)                      # KNN search for nearest downsampled point for each original point, dimension is (6, 40960, 1)
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
        
        selected_pc, selected_labels, selected_idx, cloud_ind = [],[],[],[]
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)                     # Stack lists to form matrix, dimension is (batch, nums, feature) = (6, 40960, 6)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        selected_xyz = selected_pc[:, :, 0:3]
        selected_features = selected_pc[:, :, 3:6]

        flat_inputs = self.tf_map(selected_xyz, selected_features, selected_labels, selected_idx, cloud_ind) # Return value is a list containing 24 lists

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())     # Added five lists, coordinates before each random sampling
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())    # Added five lists, coordinates of 16 neighbors of input points before each random sampling (first list has no downsampling)
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())      # Added five lists, coordinates of 16 neighbors of input points after each random sampling
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())   # Added five lists, nearest downsampled point for each original point after each random sampling

        # inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()   # Transposed
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()  # Modified, no transpose to fit later linear layer dimensions
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs


if __name__ == '__main__':

    print('Testing STPLS3D dataset...')
    
    dataset = STPLS3D('training')
    dataset_train = STPLS3DSampler(dataset, split='training')
    dataloader = DataLoader(
        dataset_train, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=dataset_train.collate_fn
    )
    
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
    
    print('\nTesting validation set...')
    dataset_val = STPLS3D('validation')
    dataset_val_sampler = STPLS3DSampler(dataset_val, split='validation')
    val_dataloader = DataLoader(
        dataset_val_sampler,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=dataset_val_sampler.collate_fn
    )
    
    for data in val_dataloader:
        print(f'Validation features shape: {data["features"].shape}')
        break
    
    print('\nDataset test complete!')
