# from open3d import linux as open3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'utils', 'nearest_neighbors'))

import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import nearest_neighbors


class ConfigSTPLS3D:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 6  # Number of valid classes
    sub_grid_size = 0.3  # preprocess_parameter

    batch_size = 3  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 20  # Number of validation steps per epoch #TODO: consider increasing to 100

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class ConfigS3DIS:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class ConfigDALES:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 8  # Number of valid classes (excluding class 0: Unknown)
    sub_grid_size = 0.25  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class DataProcessing:
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3 (search on these points)
        :param query_pts: points you want to know the neighbour index, B*N2*3 (use these points as centers for KNN)
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        if color is not None:
            color_dup = color[dup, ...]
            color_aug = np.concatenate([color, color_dup], 0)
        else:
            color_aug = None
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def metrics(confusions, ignore_unclassified=False):
        """
        Computes different metrics from confusion matrices (from KPConv code).
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :param ignore_unclassified: (bool). True if the first class should be ignored in the results
        :return: ([..., n_c] np.float32) precision, recall, F1 score, IoU score
        """

        # If the first class (often "unclassified") should be ignored, erase it from the confusion.
        if ignore_unclassified:
            confusions[..., 0, :] = 0
            confusions[..., :, 0] = 0

        # Compute TP, FP, FN. This assumes that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)    # row-wise sum
        TP_plus_FP = np.sum(confusions, axis=-2)    # column-wise sum

        # Compute precision and recall. This assumes that the second to last axis counts the truths (like the first axis of
        # a confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        REC = TP / (TP_plus_FN + 1e-6)
        PRE = TP / (TP_plus_FP + 1e-6)

        # Compute Accuracy
        ACC = np.sum(TP, axis=-1) / (np.sum(confusions, axis=(-2, -1)) + 1e-6)

        # Compute F1 score
        F1 = 2 * TP / (TP_plus_FP + TP_plus_FN + 1e-6)

        # Compute IoU
        IoU = F1 / (2 - F1)

        return PRE, REC, F1, IoU, ACC


    @staticmethod
    def get_class_weights(dataset_name, weights='none'):
        """ Get class weights for loss function. """
        
        if dataset_name == 'S3DIS':
            weighting_schemes = {
                'none': [],
                'invf': [0.00968742, 0.01122959, 0.00680682, 0.07662780, 0.08725265, 0.08761674, 0.03385965, 0.05724367, 0.04560749, 0.38098676, 0.03936018, 0.14715081, 0.01657043],
                'cb':   [0.02983311, 0.03026597, 0.02935790, 0.07464949, 0.08265889, 0.08293398, 0.04330137, 0.06017421, 0.05164507, 0.30730526, 0.04716026, 0.12820474, 0.03250976],
                'invl': [0.07007084, 0.07066960, 0.06868070, 0.07950099, 0.08017847, 0.08020038, 0.07548892, 0.07802030, 0.07690426, 0.08876493, 0.07619765, 0.08302630, 0.07229667],
                'invp': [0.06596567, 0.06694738, 0.06367832, 0.08112147, 0.08218169, 0.08221592, 0.07475934, 0.07878979, 0.07701952, 0.09523315, 0.07589320, 0.08659114, 0.06960342],
                'comf': [0.06738017, 0.06957104, 0.06062889, 0.08131651, 0.08156210, 0.08156946, 0.07876905, 0.08063356, 0.07994475, 0.08292769, 0.07940691, 0.08228308, 0.07400679]
            }
            if weights not in weighting_schemes:
                raise ValueError(f"Unknown weighting scheme '{weights}'. Choose from: {list(weighting_schemes.keys())}")
            ce_label_weight = np.array(weighting_schemes[weights], dtype=np.float32)
            return np.expand_dims(ce_label_weight, axis=0)
        
        elif dataset_name == 'DALES':
            # Classes: Ground, Vegetation, Cars, Trucks, Power lines, Fences, Poles, Buildings
            weighting_schemes = {
                'none': [],
                'invf': [0.00078, 0.00111, 0.05223, 0.18005, 0.17241, 0.09044, 0.50061, 0.00237],
                'cb'  : [0.01315, 0.01315, 0.05550, 0.17427, 0.16717, 0.09094, 0.47263, 0.01319],
                'invl': [0.09982, 0.10172, 0.12825, 0.13999, 0.13954, 0.13321, 0.15144, 0.10602],
                'invp': [0.08597, 0.08907, 0.13089, 0.14814, 0.14750, 0.13828, 0.16409, 0.09606],
                'comf': [0.07370, 0.09432, 0.14182, 0.14256, 0.14254, 0.14226, 0.14275, 0.12005]
            }
            if weights not in weighting_schemes:
                raise ValueError(f"Unknown weighting scheme '{weights}'. Choose from: {list(weighting_schemes.keys())}")
            ce_label_weight = np.array(weighting_schemes[weights], dtype=np.float32)
            return np.expand_dims(ce_label_weight, axis=0)

        elif dataset_name == 'STPLS3D':
            # Classes: ground, building, vegetation, cars, lightStreetSigns, fences
            weighting_schemes = {
                'none': [],
                'invf': [0.00572093, 0.00841892, 0.00689398, 0.14601468, 0.57806325, 0.25488824],
                'cb'  : [0.14512672, 0.14512672, 0.14512672, 0.14964478, 0.24726713, 0.16770794],
                'invl': [0.14855783, 0.15140313, 0.14991798, 0.17634655, 0.19156658, 0.18220792],
                'invp': [0.13352656, 0.1387863, 0.13604043, 0.1846123, 0.21184514, 0.19518928],
                'comf': [0.12250765, 0.14734137, 0.1356934, 0.19696381, 0.19923308, 0.19826069]
            }
            if weights not in weighting_schemes:
                raise ValueError(f"Unknown weighting scheme '{weights}'. Choose from: {list(weighting_schemes.keys())}")
            ce_label_weight = np.array(weighting_schemes[weights], dtype=np.float32)
            return np.expand_dims(ce_label_weight, axis=0)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # weight = num_per_class / float(sum(num_per_class))
        # ce_label_weight = 1 / (weight + 0.02)
        # return np.expand_dims(ce_label_weight, axis=0)
        return Y_semins
