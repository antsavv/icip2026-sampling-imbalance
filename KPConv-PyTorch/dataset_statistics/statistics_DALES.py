
from os.path import join, exists
from utils.ply import read_ply

import matplotlib.pyplot as plt
import numpy as np

class_names = ['Unknown', 'Ground', 'Vegetation', 'Cars', 'Trucks', 'Power lines', 'Fences', 'Poles', 'Buildings']
N_classes = len(class_names)


def get_full_filenames_from_txt(txt_path, ply_folder):
    """
    Read cloud names from a txt file and return full paths to ply files.
    
    Args:
        txt_path: Path to the txt file containing cloud names (one per line)
        ply_folder: Folder containing the ply files
    """
    with open(txt_path) as f:
        cloud_names = f.readlines()

    files = []
    for cloud_name in cloud_names:
        temp = cloud_name.strip()
        if temp:
            files.append(join(ply_folder, temp + '.ply'))

    return files


def process_clouds(cloud_names, base_path, header, visualize_label7=False):
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

        if visualize_label7:
            mask = data['class'] == 7
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
                ax.set_title(f"{name} - Class 7 (poles)")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.tight_layout()
                plt.savefig(f"{name}_label7_xyz.png")
                plt.close(fig)

    print('Total: ', end="")
    print(dict1)
    total_points = sum(dict1.values())
    print(f'Total points: {total_points:,}')
    print(' ')
    
    return dict1


def process_raw_dataset(base_path, visualize_label7=False):
    """Process raw (initial) DALES dataset."""
    print('\n' + '=' * 80)
    print(' RAW DATASET ')
    print('=' * 80 + '\n')
    
    raw_path = join(base_path, 'dales_ply')
    
    # Training set
    train_txt = join(raw_path, 'train.txt')
    train_folder = join(raw_path, 'train')
    if exists(train_txt) and exists(train_folder):
        train_clouds = get_full_filenames_from_txt(train_txt, train_folder)
        process_clouds(train_clouds, raw_path, 'Raw Training Set', visualize_label7)
    else:
        print(f'Raw training data not found at {raw_path}')
    
    # Testing set
    test_txt = join(raw_path, 'test.txt')
    test_folder = join(raw_path, 'test')
    if exists(test_txt) and exists(test_folder):
        test_clouds = get_full_filenames_from_txt(test_txt, test_folder)
        process_clouds(test_clouds, raw_path, 'Raw Testing Set', visualize_label7)
    else:
        print(f'Raw testing data not found at {raw_path}')


def process_downsampled_dataset(base_path, train_folder_name, test_folder_name, dataset_name, visualize_label7=False):
    """Process downsampled DALES dataset."""
    print('\n' + '=' * 80)
    print(f' {dataset_name.upper()} ')
    print('=' * 80 + '\n')
    
    raw_path = join(base_path, 'dales_ply')
    
    # Training set - use txt from raw folder, ply from downsampled folder
    train_txt = join(raw_path, 'train.txt')
    train_folder = join(base_path, train_folder_name)
    if exists(train_txt) and exists(train_folder):
        train_clouds = get_full_filenames_from_txt(train_txt, train_folder)
        # Filter to only existing files (some might not be in downsampled folder)
        train_clouds = [f for f in train_clouds if exists(f)]
        if train_clouds:
            process_clouds(train_clouds, train_folder, f'{dataset_name} Training Set', visualize_label7)
        else:
            print(f'No training ply files found in {train_folder}')
    else:
        print(f'Training data not found: txt={train_txt}, folder={train_folder}')
    
    # Testing set - use txt from raw folder, ply from downsampled folder
    test_txt = join(raw_path, 'test.txt')
    test_folder = join(base_path, test_folder_name)
    if exists(test_txt) and exists(test_folder):
        test_clouds = get_full_filenames_from_txt(test_txt, test_folder)
        # Filter to only existing files
        test_clouds = [f for f in test_clouds if exists(f)]
        if test_clouds:
            process_clouds(test_clouds, test_folder, f'{dataset_name} Testing Set', visualize_label7)
        else:
            print(f'No testing ply files found in {test_folder}')
    else:
        print(f'Testing data not found: txt={test_txt}, folder={test_folder}')


if __name__ == '__main__':

    base_path = './data/DALES/'
    visualize_label7 = False  # Set to True to visualize label 7 (poles)
    
    # Process raw (initial) dataset
    process_raw_dataset(base_path, visualize_label7)
    
    # Process downsampled datasets (KPConv style - 0.250)
    process_downsampled_dataset(
        base_path, 
        'train_val_input_0.250', 
        'test_input_0.250',
        'Downsampled 0.250 (KPConv)',
        visualize_label7
    )
    
    # Process downsampled datasets (RandLA-Net style - 0.250)
    process_downsampled_dataset(
        base_path, 
        'train_val_input_0.250_randla-net', 
        'test_input_0.250_randla-net',
        'Downsampled 0.250 (RandLA-Net)',
        visualize_label7
    )

