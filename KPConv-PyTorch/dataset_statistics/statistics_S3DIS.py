from os import listdir
from os.path import join, exists
from utils.ply import read_ply

import matplotlib.pyplot as plt
import numpy as np

class_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter']

N_classes = len(class_names)

train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
test_areas = ['Area_5']


def process_clouds(cloud_names, header, visualize_label10=False):
    print(6 * N_classes * '-' + f' {header} ' + 6 * N_classes * '-')
    dict1 = {i: 0 for i in range(N_classes)}

    for cloud_name in cloud_names:
        data = read_ply(cloud_name)
        unique, counts = np.unique(data['class'], return_counts=True)
        # Convert numpy int64 to Python int for cleaner printing
        dict2 = {int(k): int(v) for k, v in zip(unique, counts)}

        # Extract just the filename for display
        name = cloud_name.split('/')[-1].replace('.ply', '')
        print(name + ': ', end="")
        print(dict2)

        for key in dict1:
            if key in dict2:
                dict1[key] += dict2[key]

        if visualize_label10:
            mask = data['class'] == 10
            if np.any(mask): # sanity check
                x = data['x'][mask]
                y = data['y'][mask]
                z = data['z'][mask]
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(projection='3d')
                ax.scatter(x, y, z, s=1, c='red')
                ax.view_init(elev=90, azim=-90)  # XY plane
                ax.set_xlim([data['x'].min(), data['x'].max()])
                ax.set_ylim([data['y'].min(), data['y'].max()])
                ax.set_title(f"{name} - Class 10 (sofa)")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.tight_layout()
                plt.savefig(f"{name}_label10_xyz.png")
                plt.close(fig)

    print('Total: ', end="")
    print(dict1)
    total_points = sum(dict1.values())
    print(f'Total points: {total_points:,}')
    print(' ')
    
    return dict1


def get_cloud_names_by_area(ply_path, area_names):
    """Get all room ply files for given areas (e.g., Area_1_WC_1.ply, Area_1_conferenceRoom_1.ply)"""
    all_files = listdir(ply_path)
    cloud_names = []
    for area in area_names:
        # Match files starting with Area_X_ and ending with .ply
        area_files = [join(ply_path, f) for f in all_files if f.startswith(area + '_') and f.endswith('.ply')]
        cloud_names.extend(sorted(area_files))
    return cloud_names


def get_area_files(ply_path, area_names):
    """Get Area_X.ply files for given areas (whole area files)"""
    cloud_names = []
    for area in area_names:
        file_path = join(ply_path, area + '.ply')
        if exists(file_path):
            cloud_names.append(file_path)
    return cloud_names


def process_raw_dataset(base_path, visualize_label10=False):
    """Process raw (initial) S3DIS dataset."""
    print('\n' + '=' * 80)
    print(' RAW DATASET ')
    print('=' * 80 + '\n')
    
    raw_path = join(base_path, 'original_ply')
    
    if exists(raw_path):
        train_clouds = get_area_files(raw_path, train_areas)
        test_clouds = get_area_files(raw_path, test_areas)
        
        if train_clouds:
            process_clouds(train_clouds, 'Raw Training Set', visualize_label10)
        else:
            print(f'No training ply files found in {raw_path}')
            
        if test_clouds:
            process_clouds(test_clouds, 'Raw Testing Set', visualize_label10)
        else:
            print(f'No testing ply files found in {raw_path}')
    else:
        print(f'Raw data not found at {raw_path}')


def process_downsampled_dataset(base_path, folder_name, dataset_name, use_room_files=False, visualize_label10=False):
    """Process downsampled S3DIS dataset.
    
    Args:
        base_path: Base path to the dataset
        folder_name: Name of the downsampled folder
        dataset_name: Display name for the dataset
        use_room_files: If True, look for room-level files (Area_X_room_N.ply), 
                        else look for area files (Area_X.ply)
        visualize_label10: If True, visualize class 10 (sofa) points
    """
    print('\n' + '=' * 80)
    print(f' {dataset_name.upper()} ')
    print('=' * 80 + '\n')
    
    ply_path = join(base_path, folder_name)
    
    if exists(ply_path):
        if use_room_files:
            train_clouds = get_cloud_names_by_area(ply_path, train_areas)
            test_clouds = get_cloud_names_by_area(ply_path, test_areas)
        else:
            train_clouds = get_area_files(ply_path, train_areas)
            test_clouds = get_area_files(ply_path, test_areas)
        
        if train_clouds:
            process_clouds(train_clouds, f'{dataset_name} Training Set', visualize_label10)
        else:
            print(f'No training ply files found in {ply_path}')
            
        if test_clouds:
            process_clouds(test_clouds, f'{dataset_name} Testing Set', visualize_label10)
        else:
            print(f'No testing ply files found in {ply_path}')
    else:
        print(f'Data not found at {ply_path}')


if __name__ == '__main__':
    base_path = './data/S3DIS-aligned-version/'
    visualize_label10 = False  # Set to True to visualize label 10 (sofa)

    # Process raw (initial) dataset
    process_raw_dataset(base_path, visualize_label10)
    
    # Process downsampled datasets (KPConv style - 0.030)
    process_downsampled_dataset(
        base_path,
        'input_0.030',
        'Downsampled 0.030 (KPConv)',
        use_room_files=False,
        visualize_label10=visualize_label10
    )
    
    # Process downsampled datasets (RandLA-Net style - 0.040)
    process_downsampled_dataset(
        base_path,
        'input_0.040_randla-net',
        'Downsampled 0.040 (RandLA-Net)',
        use_room_files=True,
        visualize_label10=visualize_label10
    )