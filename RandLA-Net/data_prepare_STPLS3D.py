"""
Data preparation script for STPLS3D dataset with RandLA-Net.

This script preprocesses the STPLS3D point cloud dataset for training with RandLA-Net.
It performs:
1. Grid subsampling of point clouds
2. KDTree construction for fast neighbor search
3. Projection index computation for validation/test evaluation

Usage:
    python data_prepare_STPLS3D.py

Expected input structure:
    data/STPLS3D/
    ├── original_ply/           # Original PLY files with XYZ, RGB, class (with classes remapped to 0-5; this was done separately)
    ├── train.txt               # List of training cloud names (without .ply)
    └── test.txt                # List of test cloud names (without .ply)

Output structure:
    data/STPLS3D/
    └── input_{grid_size}_randla-net/
        ├── *.ply               # Subsampled point clouds
        ├── *_KDTree.pkl        # KDTree structures
        └── *_proj.pkl          # Projection indices (for validation clouds)
"""

from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os
import sys
import pickle

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from helper_ply import write_ply, read_ply
from helper_tool import DataProcessing as DP

# Configuration
dataset_path = 'data/STPLS3D'
sub_grid_size = 0.3

# Output folder
output_folder = join(dataset_path, 'input_{:.3f}_randla-net'.format(sub_grid_size))
os.makedirs(output_folder, exist_ok=True)

# Input folders
original_ply_path = join(dataset_path, 'original_ply')

# Read file lists
with open(join(dataset_path, 'train.txt'), 'r') as f:
    train_files = [line.strip() for line in f.readlines() if line.strip()]
with open(join(dataset_path, 'test.txt'), 'r') as f:
    test_files = [line.strip() for line in f.readlines() if line.strip()]

# Validation split - using same as KPConv (WMSC_points is the test file)
validation_files = test_files.copy()


def prepare_stpls3d_data():
    """
    Preprocess STPLS3D dataset for RandLA-Net. Creates subsampled point clouds and KDTree structures.
    """
    
    all_files = train_files + test_files
    # Remove duplicates while preserving order
    all_files = list(dict.fromkeys(all_files))
    
    print(f'\n=== STPLS3D Data Preparation ===')
    print(f'Total files to process: {len(all_files)}')
    print(f'Training files: {len(train_files)}')
    print(f'Test/Validation files: {len(test_files)}')
    print(f'Grid size: {sub_grid_size}')
    print(f'Output folder: {output_folder}')
    print()
    
    total_class_counts = {}
    
    print('=== Processing Point Clouds ===')
    for i, cloud_name in enumerate(all_files):
        print(f'\nProcessing file {i+1}/{len(all_files)}: {cloud_name}')
        
        original_ply = join(original_ply_path, cloud_name + '.ply')
        output_ply = join(output_folder, cloud_name + '.ply')
        kdtree_file = join(output_folder, cloud_name + '_KDTree.pkl')
        
        # Check if source file exists
        if not exists(original_ply):
            print(f'  WARNING: Source file not found: {original_ply}')
            continue
        
        # Skip if already processed
        if exists(kdtree_file) and exists(output_ply):
            print(f'  Already processed, skipping...')
            
            # Still count classes from existing file
            data = read_ply(output_ply)
            sub_labels = data['class']
            unique, counts = np.unique(sub_labels, return_counts=True)
            for label, count in zip(unique, counts):
                if label not in total_class_counts:
                    total_class_counts[label] = 0
                total_class_counts[label] += count
            continue
        
        # Read original point cloud
        data = read_ply(original_ply)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.vstack((data['red'], data['green'], data['blue'])).T
        labels = data['class']
        
        print(f'  Original points: {points.shape[0]}')
        print(f'  Label range: {labels.min()} - {labels.max()}')
        
        # Subsample the point cloud
        sub_points, sub_colors, sub_labels = DP.grid_sub_sampling(
            points.astype(np.float32),
            features=colors.astype(np.float32),
            labels=labels.astype(np.int32),
            grid_size=sub_grid_size
        )
        
        # Squeeze labels and normalize colors
        sub_labels = np.squeeze(sub_labels)
        sub_colors = sub_colors / 255.0  # Normalize to [0, 1]
        
        print(f'  Subsampled points: {sub_points.shape[0]}')
        
        # Count classes
        unique, counts = np.unique(sub_labels, return_counts=True)
        for label, count in zip(unique, counts):
            if label not in total_class_counts:
                total_class_counts[label] = 0
            total_class_counts[label] += count
        
        # Create KDTree for fast nearest neighbor search
        search_tree = KDTree(sub_points)
        
        # Save subsampled point cloud
        write_ply(output_ply, 
                  [sub_points, sub_colors, sub_labels.astype(np.int32)], 
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
        
        # Save KDTree
        with open(kdtree_file, 'wb') as f:
            pickle.dump(search_tree, f)
        
        print(f'  Saved to {output_ply}')
    
    # Generate projection indices for validation/test files
    print('\n=== Generating Projection Indices ===')
    
    for cloud_name in validation_files:
        print(f'\nGenerating projection for: {cloud_name}')
        
        original_ply = join(original_ply_path, cloud_name + '.ply')
        proj_file = join(output_folder, cloud_name + '_proj.pkl')
        kdtree_file = join(output_folder, cloud_name + '_KDTree.pkl')
        
        # Check if source exists
        if not exists(original_ply):
            print(f'  WARNING: Source file not found: {original_ply}')
            continue
            
        # Skip if already exists
        if exists(proj_file):
            print(f'  Already exists, skipping...')
            continue
        
        # Read original point cloud
        data_original = read_ply(original_ply)
        points_original = np.vstack((data_original['x'], data_original['y'], data_original['z'])).T
        labels_original = data_original['class']
        
        print(f'  Original points: {points_original.shape[0]}')
        
        # Load KDTree
        with open(kdtree_file, 'rb') as f:
            search_tree = pickle.load(f)
        
        # Find nearest subsampled point for each original point
        proj_idx = np.squeeze(search_tree.query(points_original, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        
        # Save projection indices and labels
        with open(proj_file, 'wb') as f:
            pickle.dump([proj_idx, labels_original], f)
        
        print(f'  Saved projection indices ({len(proj_idx)} points)')
    
    print('\n=== Data Preparation Complete ===')
    print(f'Training files: {len(train_files)}')
    print(f'Test/Validation files: {len(test_files)}')
    print(f'Subsampling grid size: {sub_grid_size}')
    print(f'\nClass distribution (subsampled):')
    
    # STPLS3D classes
    class_names = {
        0: 'ground',
        1: 'building', 
        2: 'vegetation',
        3: 'cars',
        4: 'lightStreetSigns',
        5: 'fences'
    }
    
    cls_num_list = []
    for label in sorted(total_class_counts.keys()):
        count = total_class_counts[label]
        name = class_names.get(label, f'class_{label}')
        print(f'  {label}: {name:20s} - {count:,} points')
        cls_num_list.append(count)
    
    print(f'\ncls_num_list for train_randlanet.py:')
    print(f'  {cls_num_list}')


if __name__ == '__main__':
    prepare_stpls3d_data()