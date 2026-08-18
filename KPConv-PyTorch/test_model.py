#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start testing on DALES datataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import argparse
import os
import random

from datasets.DALES import *
from datasets.S3DIS import *
from datasets.STPLS3D import *

from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPFCNN

# Function for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Use True when input is fixed


if __name__ == '__main__':

    setup_seed(42)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument('--chosen_log', type=str, required=True, help="Path to the chosen log directory")
    parser.add_argument('--gpu', type=str, default='0', help="GPU ID to use (default: 0)")
    args = parser.parse_args()

    # Assign arguments to variables
    chosen_log = args.chosen_log
    GPU_ID = args.gpu

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    # config.augment_noise = 0.0001
    # config.augment_symmetries = False
    # config.batch_num = 3
    # config.in_radius = 4
    config.do_augmentations = False
    
    config.validation_size = 200
    config.input_threads = 4

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initiate dataset
    if config.dataset.startswith('DALES'):
        test_dataset = DALESDataset(config, set='test', use_potentials=True)
        test_sampler = DALESSampler(test_dataset)
        collate_fn = DALESCollate
    elif config.dataset == 'S3DIS':
        test_dataset = S3DISDataset(config, set='test', use_potentials=True)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif config.dataset == 'STPLS3D':
        test_dataset = STPLS3DDataset(config, set='test', use_potentials=True)
        test_sampler = STPLS3DSampler(test_dataset)
        collate_fn = STPLS3DCollate
    else:
        raise ValueError('Unsupported dataset: ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True,
                             persistent_workers=True)

    try:
        # Calibrate samplers
        test_sampler.calibration(test_loader, verbose=True)

        print('\nModel Preparation')
        print('*****************')

        # Define network model (assuming cloud_segmentation task)
        t1 = time.time()
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

        # Define a tester class
        tester = ModelTester(net, chkp_path=chosen_chkp)
        print('Done in {:.1f}s\n'.format(time.time() - t1))

        print('\nStart test')
        print('**********\n')

        start_time = time.time()
        
        # Use task-specific test function
        if config.dataset_task == 'slam_segmentation':
            tester.slam_segmentation_test(net, test_loader, config)
        else:
            tester.cloud_segmentation_test(net, test_loader, config)
        
        test_time = time.time() - start_time

        
        hours, rem = divmod(test_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Test completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)\n")

    finally:
        # Ensure DataLoader workers are cleaned up
        del test_loader