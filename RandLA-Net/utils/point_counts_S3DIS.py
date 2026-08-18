from os.path import join
from helper_ply import read_ply
import glob
import numpy as np

# RandLA-Net class ordering
class_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

N_classes = len(class_names)


def process_clouds(cloud_names, ply_path, header):
    print(6 * N_classes * '-' + f' {header} ' + 6 * N_classes * '-')
    
    # Group clouds by area
    area_clouds = {}
    for cloud_name in cloud_names:
        area = cloud_name.split('/')[-1].split('_')[0] + '_' + cloud_name.split('/')[-1].split('_')[1]
        if area not in area_clouds:
            area_clouds[area] = []
        area_clouds[area].append(cloud_name)
    
    # Process each area
    total_dict = {i: 0 for i in range(N_classes)}
    
    for area in sorted(area_clouds.keys()):
        area_dict = {i: 0 for i in range(N_classes)}
        
        for cloud_name in area_clouds[area]:
            data = read_ply(cloud_name)
            unique, counts = np.unique(data['class'], return_counts=True)
            dict2 = {int(k): int(v) for k, v in zip(unique, counts)}
            
            for key in area_dict:
                if key in dict2:
                    area_dict[key] += dict2[key]
        
        print(f'{area}: {area_dict}')
        
        for key in total_dict:
            total_dict[key] += area_dict[key]
    
    print(f'Total: {total_dict}')
    print(' ')
    
    # Print class breakdown with names
    print('Class breakdown:')
    for i, class_name in enumerate(class_names):
        print(f'{i}: {class_name:12s} - {total_dict[i]:,} points')
    print(' ')


if __name__ == '__main__':
    ply_path = './data/S3DIS-aligned-version/original_ply_randla-net/'

    # Get all ply files for each area
    train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    test_areas = ['Area_5']
    
    train_cloud_names = []
    for area in train_areas:
        train_cloud_names.extend(glob.glob(join(ply_path, area + '_*.ply')))
    train_cloud_names.sort()
    
    test_cloud_names = []
    for area in test_areas:
        test_cloud_names.extend(glob.glob(join(ply_path, area + '_*.ply')))
    test_cloud_names.sort()

    print(f'Found {len(train_cloud_names)} training clouds')
    print(f'Found {len(test_cloud_names)} test clouds')
    print()

    process_clouds(train_cloud_names, ply_path, 'RandLA-Net Training Set')
    process_clouds(test_cloud_names, ply_path, 'RandLA-Net Testing Set')
