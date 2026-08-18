#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
# ----------------------------------------------------------------------------------------------------------------------
#      Callable script to start a training on DALES dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os

import sys
import time
import torch
import numpy as np
import random
import argparse

# Dataset
from datasets.DALES import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN

import gc


# Function for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Use True when input is fixed


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


class DALESConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'DALES'

    # Number of classes in the dataset (This value is overwritten by dataset class when initializing dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 4

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary'
                    ]

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter (increase value to reduce memory cost)
    first_subsampling_dl = 0.250  # 0.03

    # Radius of the input sphere (decrease value to reduce memory cost)
    in_radius = 30 * first_subsampling_dl

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 5.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.0

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128
    in_features_dim = 1

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 400

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch (decrease to reduce memory cost, but it should remain > 3 for stability)
    batch_num = 10

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    do_augmentations = True
    augment_debug = False
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_color = 0.8

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'
  
    # Class weighting schemes for DALES (8 classes), normalised to sum to 1.
    # Derived from the full-resolution class point counts (cls_num_list below)
    # with dataset_statistics/calculate_weights.py
    weighting_schemes = {
        'none': [],
        'invf': [0.00078, 0.00111, 0.05223, 0.18005, 0.17241, 0.09044, 0.50061, 0.00237],
        'cb'  : [0.01315, 0.01315, 0.05550, 0.17427, 0.16717, 0.09094, 0.47263, 0.01319],
        'invl': [0.09982, 0.10172, 0.12825, 0.13999, 0.13954, 0.13321, 0.15144, 0.10602],
        'invp': [0.08597, 0.08907, 0.13089, 0.14814, 0.14750, 0.13828, 0.16409, 0.09606],
        'comf': [0.07370, 0.09432, 0.14182, 0.14256, 0.14254, 0.14226, 0.14275, 0.12005],
    }

    # Active class weights, set from the --weights command line argument
    class_w = []

    # Choose loss function (loss_function)
    # Options: cross_entropy, ldam_loss, ladj_loss, focal_loss, seesaw_loss, balanced_softmax

    focal_loss_alpha = 1.0
    focal_loss_mode = 'normal'  # 'normal', 'quantile', 'non-deterministic'
    focal_loss_gamma = 1.0

    cls_num_list = np.array([171803872, 120576463, 2567061, 744729, 777716, 1482534, 267847, 56648714])
    ldam_loss_max_m = 0.7
    ldam_loss_s = 1.0
    
    ladju_tau = 0.3

    # seesaw_loss_p = 0.8
    # seesaw_loss_q = 2.0
    seesaw_loss_eps = 1e-2

    # Do we need to save convergence
    saving = True
    saving_path = None


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Parse command line arguments
    ############################

    parser = argparse.ArgumentParser(description='Train KPConv on DALES dataset')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (default: 0)')
    parser.add_argument('--weights', type=str, default='none',
                        choices=['none', 'invf', 'cb', 'invl', 'invp', 'comf'],
                        help='Class weighting scheme to use (default: none, i.e. uniform)')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'ldam_loss', 'ladj_loss', 'focal_loss', 'seesaw_loss', 'balanced_softmax'],
                        help='Loss function to use (default: cross_entropy)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--seesaw_p', type=float, default=0.2, help='Seesaw loss p parameter (default: 0.2)')
    parser.add_argument('--seesaw_q', type=float, default=2.0, help='Seesaw loss q parameter (default: 2.0)')
    args = parser.parse_args()


    # Set up random seed for reproducibility
    setup_seed(args.seed)

    # Set which gpu is going to be used
    GPU_ID = args.gpu

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    torch.cuda.empty_cache()
    gc.collect()

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    previous_training_path = ''

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results/DALES', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results/DALES', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = DALESConfig()
    
    # Set loss function from command line argument
    config.loss_function = args.loss

    # Set class weighting scheme from command line argument
    config.class_w = config.weighting_schemes[args.weights]

    # Set seesaw loss parameters from command line arguments
    config.seesaw_loss_p = args.seesaw_p
    config.seesaw_loss_q = args.seesaw_q
    
    if previous_training_path:
        config.load(os.path.join('results/DALES', previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    # if len(sys.argv) > 1:
    #     config.saving_path = sys.argv[1]
    
    # Create folder name with seesaw loss parameters if using seesaw loss
    if args.loss == 'seesaw_loss':
        config.saving_path = time.strftime('results/DALES/DALES-Log_%Y-%m-%d_%H-%M-%S_w_' + args.weights + '_' + args.loss + '_p' + str(args.seesaw_p) + '_q' + str(args.seesaw_q) + '_seed_' + str(args.seed), time.gmtime())
    else:
        config.saving_path = time.strftime('results/DALES/DALES-Log_%Y-%m-%d_%H-%M-%S_w_' + args.weights + '_' + args.loss + '_seed_' + str(args.seed), time.gmtime())

    # Initialize datasets
    training_dataset = DALESDataset(config, set='training', use_potentials=True)
    
    val_dataset = DALESDataset(config, set='validation', use_potentials=True)

    # Initialize samplers
    training_sampler = DALESSampler(training_dataset)
    val_sampler = DALESSampler(val_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=DALESCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True,
                                 persistent_workers=True)

    test_loader = DataLoader(val_dataset,
                             batch_size=1,
                             sampler=val_sampler,
                             collate_fn=DALESCollate,
                             num_workers=config.input_threads,
                             pin_memory=True,
                             persistent_workers=True)

    try:

        # Calibrate samplers
        training_sampler.calibration(training_loader, verbose=True)
        val_sampler.calibration(test_loader, verbose=True)

        # Optional debug functions
        # debug_timing(training_dataset, training_loader)
        # debug_timing(val_dataset, test_loader)
        # debug_upsampling(training_dataset, training_loader)
        # debug_show_clouds(training_dataset, training_loader)

        print('\nModel Preparation')
        print('*****************')

        # Define network model
        t1 = time.time()
        net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)

        debug = False
        if debug:
            print('\n*************************************\n')
            print(net)
            print('\n*************************************\n')
            for param in net.parameters():
                if param.requires_grad:
                    print(param.shape)
            print('\n*************************************\n')
            print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
            print('\n*************************************\n')

        # Define a trainer class
        trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
        print('Done in {:.1f}s\n'.format(time.time() - t1))

        print('\nStart training')
        print('**************')

        # Training
        trainer.train(net, training_loader, test_loader, config)

    finally:
        del training_loader
        del test_loader

    print('Forcing exit now')
    sys.exit(0)
