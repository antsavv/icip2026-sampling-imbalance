from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle
import re
import warnings

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

dataset_path = 'data/S3DIS-aligned-version'
anno_paths = [line.rstrip() for line in open(join(BASE_DIR, 'utils/meta/anno_paths.txt'))]
anno_paths = [join(dataset_path, p) for p in anno_paths]

gt_class = [x.rstrip() for x in open(join(BASE_DIR, 'utils/meta/class_names.txt'))]
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}

sub_grid_size = 0.04
original_pc_folder = join(dataset_path, 'original_ply_randla-net')
sub_pc_folder = join(dataset_path, 'input_{:.3f}_randla-net'.format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = '.ply'


def fix_malformed_file(filepath):
    """
    Fix non-ASCII/control characters in S3DIS annotation files.
    Known issue in Area_5/hallway_6/Annotations/ceiling_1.txt
    """
    pattern = r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F-\x9F]'
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed = False
    for l_i, line in enumerate(lines):
        if re.search(pattern, line):
            print(f'Line {l_i} in {filepath} contains non-ASCII characters. Fixing...')
            # Check if removing chars or replacing with space is needed
            ref_n = len(lines[l_i - 1].strip().split()) if l_i > 0 else 6
            cur_n = len(lines[l_i].strip().split())
            lines[l_i] = re.sub(pattern, '' if cur_n == ref_n else ' ', line)
            fixed = True
    
    if fixed:
        with open(filepath, 'w') as f:
            f.writelines(lines)
    
    return fixed


def convert_pc2ply(anno_path, save_path):
    """
    Convert original dataset files to ply file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    :param anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
    :param save_path: path to save original point clouds (each line is XYZRGBL)
    :return: None
    """
    data_list = []

    for f in glob.glob(join(anno_path, '*.txt')):
        class_name = os.path.basename(f).split('_')[0]
        if class_name not in gt_class:  # note: in some room there is 'stairs' class..
            class_name = 'clutter'
        
        # Try to read the file, fix if malformed
        try:
            pc = pd.read_csv(f, header=None, sep='\s+').values
        except Exception as e:
            warnings.warn(f'Error reading {f}: {e}. Attempting to fix...')
            fix_malformed_file(f)
            pc = pd.read_csv(f, header=None, sep='\s+').values
        
        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
        data_list.append(np.concatenate([pc, labels], 1))  # Nx7

    pc_label = np.concatenate(data_list, 0)
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    xyz = pc_label[:, :3].astype(np.float32)
    colors = pc_label[:, 3:6].astype(np.uint8)
    labels = pc_label[:, 6].astype(np.uint8)
    write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


if __name__ == '__main__':
    # Automatic handling of malformed files (e.g., Area_5/hallway_6/Annotations/ceiling_1.txt)
    for annotation_path in anno_paths:
        print(annotation_path)
        elements = str(annotation_path).split('/')
        out_file_name = elements[-3] + '_' + elements[-2] + out_format
        convert_pc2ply(annotation_path, join(original_pc_folder, out_file_name))
