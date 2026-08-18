from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os, sys, pickle

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply, read_ply
from helper_tool import DataProcessing as DP

dataset_path = 'data/DALES'
sub_grid_size = 0.25

train_val_folder = join(dataset_path, 'train_val_input_{:.3f}_randla-net'.format(sub_grid_size))
test_folder = join(dataset_path, 'test_input_{:.3f}_randla-net'.format(sub_grid_size))
os.makedirs(train_val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

ply_path = join(dataset_path, 'dales_ply')
with open(join(ply_path, 'train.txt'), 'r') as f:
    train_files = [line.strip() for line in f.readlines()]
with open(join(ply_path, 'test.txt'), 'r') as f:
    test_files = [line.strip() for line in f.readlines()]


def prepare_dales_data():
    """
    Preprocess DALES dataset for RandLA-Net.
    Creates subsampled point clouds and KDTree structures.
    """
    
    # Process training files
    print('\n=== Processing Training Files ===')
    for i, cloud_name in enumerate(train_files):
        print(f'\nProcessing training file {i+1}/{len(train_files)}: {cloud_name}')
        
        original_ply = join(ply_path, 'train', cloud_name + '.ply')
        output_ply = join(train_val_folder, cloud_name + '.ply')
        kdtree_file = join(train_val_folder, cloud_name + '_KDTree.pkl')
        
        # Skip if already processed
        if exists(kdtree_file):
            print(f'  Already processed, skipping...')
            continue
        
        # Read original point cloud
        data = read_ply(original_ply)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        labels = data['class']
        
        print(f'  Original points: {points.shape[0]}')
        
        # Subsample the point cloud
        sub_points, sub_labels = DP.grid_sub_sampling(
            points.astype(np.float32),
            labels=labels.astype(np.int32),
            grid_size=sub_grid_size
        )
        
        # Squeeze labels
        sub_labels = np.squeeze(sub_labels)
        
        print(f'  Subsampled points: {sub_points.shape[0]}')
        
        # Create KDTree for fast nearest neighbor search
        search_tree = KDTree(sub_points)
        
        # Save subsampled point cloud
        write_ply(output_ply, [sub_points, sub_labels], ['x', 'y', 'z', 'class'])
        
        # Save KDTree
        with open(kdtree_file, 'wb') as f:
            pickle.dump(search_tree, f)
        
        print(f'  Saved to {output_ply}')
    
    # Process test files
    print('\n=== Processing Test Files ===')
    for i, cloud_name in enumerate(test_files):
        print(f'\nProcessing test file {i+1}/{len(test_files)}: {cloud_name}')
        
        original_ply = join(ply_path, 'test', cloud_name + '.ply')
        output_ply = join(test_folder, cloud_name + '.ply')
        kdtree_file = join(test_folder, cloud_name + '_KDTree.pkl')
        
        # Skip if already processed
        if exists(kdtree_file):
            print(f'  Already processed, skipping...')
            continue
        
        # Read original point cloud
        data = read_ply(original_ply)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        labels = data['class']
        
        print(f'  Original points: {points.shape[0]}')
        
        # Subsample the point cloud
        sub_points, sub_labels = DP.grid_sub_sampling(
            points.astype(np.float32),
            labels=labels.astype(np.int32),
            grid_size=sub_grid_size
        )
        
        # Squeeze labels
        sub_labels = np.squeeze(sub_labels)
        
        print(f'  Subsampled points: {sub_points.shape[0]}')
        
        # Create KDTree for fast nearest neighbor search
        search_tree = KDTree(sub_points)
        
        # Save subsampled point cloud
        write_ply(output_ply, [sub_points, sub_labels], ['x', 'y', 'z', 'class'])
        
        # Save KDTree
        with open(kdtree_file, 'wb') as f:
            pickle.dump(search_tree, f)
        
        print(f'  Saved to {output_ply}')
    
    # Generate projection indices for validation and test sets
    print('\n=== Generating Projection Indices ===')
    
    # Validation split indices (same as KPConv)
    validation_split = [0, 9, 13, 18, 22, 25]
    
    for i, cloud_name in enumerate(train_files):
        if i in validation_split:
            print(f'\nGenerating projection for validation file: {cloud_name}')
            
            original_ply = join(ply_path, 'train', cloud_name + '.ply')
            sub_ply = join(train_val_folder, cloud_name + '.ply')
            proj_file = join(train_val_folder, cloud_name + '_proj.pkl')
            
            # Skip if already exists
            if exists(proj_file):
                print(f'  Already exists, skipping...')
                continue
            
            # Read original and subsampled clouds
            data_original = read_ply(original_ply)
            points_original = np.vstack((data_original['x'], data_original['y'], data_original['z'])).T
            labels_original = data_original['class']
            
            # Load KDTree
            kdtree_file = join(train_val_folder, cloud_name + '_KDTree.pkl')
            with open(kdtree_file, 'rb') as f:
                search_tree = pickle.load(f)
            
            # Find nearest subsampled point for each original point
            proj_idx = np.squeeze(search_tree.query(points_original, return_distance=False))
            
            # Save projection indices and labels
            with open(proj_file, 'wb') as f:
                pickle.dump([proj_idx, labels_original], f)
            
            print(f'  Saved projection indices')
    
    # Generate projection indices for test files
    for cloud_name in test_files:
        print(f'\nGenerating projection for test file: {cloud_name}')
        
        original_ply = join(ply_path, 'test', cloud_name + '.ply')
        sub_ply = join(test_folder, cloud_name + '.ply')
        proj_file = join(test_folder, cloud_name + '_proj.pkl')
        
        # Skip if already exists
        if exists(proj_file):
            print(f'  Already exists, skipping...')
            continue
        
        # Read original and subsampled clouds
        data_original = read_ply(original_ply)
        points_original = np.vstack((data_original['x'], data_original['y'], data_original['z'])).T
        labels_original = data_original['class']
        
        # Load KDTree
        kdtree_file = join(test_folder, cloud_name + '_KDTree.pkl')
        with open(kdtree_file, 'rb') as f:
            search_tree = pickle.load(f)
        
        # Find nearest subsampled point for each original point
        proj_idx = np.squeeze(search_tree.query(points_original, return_distance=False))
        
        # Save projection indices and labels
        with open(proj_file, 'wb') as f:
            pickle.dump([proj_idx, labels_original], f)
        
        print(f'  Saved projection indices')
    
    print('\n=== Data Preparation Complete ===')
    print(f'Training/Validation files: {len(train_files)}')
    print(f'Test files: {len(test_files)}')
    print(f'Subsampling grid size: {sub_grid_size}')


if __name__ == '__main__':
    prepare_dales_data()
