"""
Unified testing script for RandLA-Net.
"""

import numpy as np
import os, argparse, time, warnings

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

from helper_tool import DataProcessing as DP
from helper_ply import write_ply

from RandLANet import Network, compute_loss, IoUCalculator


# ================================================================================
# Dataset-specific configurations
# ================================================================================

DATASET_CONFIG = {
    'S3DIS': {
        'config_class': 'ConfigS3DIS',
        'dataset_class': 'S3DIS',
        'sampler_class': 'S3DISSampler',
        'log_dir': 'test/S3DIS',
        'has_ignored_labels': False,
    },
    'DALES': {
        'config_class': 'ConfigDALES',
        'dataset_class': 'DALES',
        'sampler_class': 'DALESSampler',
        'log_dir': 'test/DALES',
        'has_ignored_labels': True,  # Class 0 is ignored, network outputs 0-7 map to labels 1-8
    },
    'STPLS3D': {
        'config_class': 'ConfigSTPLS3D',
        'dataset_class': 'STPLS3D',
        'sampler_class': 'STPLS3DSampler',
        'log_dir': 'test/STPLS3D',
        'has_ignored_labels': False,  # No ignored labels in STPLS3D
    },
}


def get_dataset_components(dataset_name):
    """Dynamically import and return dataset-specific components."""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")
    
    config = DATASET_CONFIG[dataset_name]
    
    # Import config from helper_tool
    from helper_tool import ConfigS3DIS, ConfigDALES, ConfigSTPLS3D
    config_map = {
        'ConfigS3DIS': ConfigS3DIS,
        'ConfigDALES': ConfigDALES,
        'ConfigSTPLS3D': ConfigSTPLS3D,
    }
    cfg = config_map[config['config_class']]
    
    # Import dataset and sampler
    if dataset_name == 'S3DIS':
        from datasets.S3DIS import S3DIS, S3DISSampler
        return cfg, S3DIS, S3DISSampler
    
    elif dataset_name == 'DALES':
        from datasets.DALES import DALES, DALESSampler
        return cfg, DALES, DALESSampler
    
    elif dataset_name == 'STPLS3D':
        from datasets.STPLS3D import STPLS3D, STPLS3DSampler
        return cfg, STPLS3D, STPLS3DSampler
    
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test RandLA-Net on various datasets')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='S3DIS',
                        choices=['S3DIS', 'DALES', 'STPLS3D'],
                        help='Dataset to test on [default: S3DIS]')
    
    # Common arguments
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use [default: 0]')
    parser.add_argument('--num_votes', type=int, default=10, help='Number of voting passes [default: 10]')
    
    # Dataset-specific arguments
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test (S3DIS only), option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='test', choices=['validation', 'test'], help='Test on validation or test set (DALES only) [default: test]')
    
    return parser.parse_args()


FLAGS = parse_args()

# Get dataset components
dataset_config = DATASET_CONFIG[FLAGS.dataset]
cfg, dataset_class, sampler_class = get_dataset_components(FLAGS.dataset)

#################################################   log   #################################################

LOG_DIR = dataset_config['log_dir']

# Extract training log directory name from checkpoint path
if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
    train_log_name = os.path.basename(os.path.dirname(FLAGS.checkpoint_path))
    LOG_DIR = os.path.join(LOG_DIR, train_log_name)
else:
    LOG_DIR = os.path.join(LOG_DIR, time.strftime(f'{FLAGS.dataset}-%Y-%m-%d_%H-%M-%S', time.gmtime()))

# Determine output folder name and log file name based on dataset
if FLAGS.dataset == 'S3DIS':
    preds_folder = 'val_preds'
    log_file_name = f'log_test_Area_{FLAGS.test_area:d}.txt'
    test_mode = 'validation'

elif FLAGS.dataset == 'DALES':
    preds_folder = f'{FLAGS.mode}_preds'
    log_file_name = f'log_test_{FLAGS.mode}.txt'
    test_mode = FLAGS.mode

elif FLAGS.dataset == 'STPLS3D':
    preds_folder = f'{FLAGS.mode}_preds'
    log_file_name = f'log_test_{FLAGS.mode}.txt'
    test_mode = FLAGS.mode

if not os.path.exists(LOG_DIR):
    os.makedirs(os.path.join(LOG_DIR, preds_folder))

LOG_FOUT = open(os.path.join(LOG_DIR, log_file_name), 'a')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


#################################################   dataset   #################################################

# Create dataset and dataloader
if FLAGS.dataset == 'S3DIS':
    dataset = dataset_class(FLAGS.test_area)
    test_dataset = sampler_class(dataset, 'validation')

elif FLAGS.dataset == 'DALES':
    dataset = dataset_class(FLAGS.mode)
    test_dataset = sampler_class(dataset, FLAGS.mode)

elif FLAGS.dataset == 'STPLS3D':
    dataset = dataset_class(FLAGS.mode)
    test_dataset = sampler_class(dataset, FLAGS.mode)

test_dataloader = DataLoader(test_dataset, batch_size=cfg.val_batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
    device = torch.device('cpu')

net = Network(cfg)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

checkpoint_path = FLAGS.checkpoint_path
print(os.path.isfile(checkpoint_path))
if checkpoint_path is not None and os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model restored from %s" % checkpoint_path)
else:
   raise ValueError('CheckPointPathError')



    
#################################################   test function   ###########################################
class ModelTester:
    def __init__(self, dataset, test_mode):
        """
        Initialize the tester with prediction arrays for each cloud.
        
        Args:
            dataset: The dataset object
            test_mode: 'validation' for S3DIS, or 'validation'/'test' for DALES
        """
        self.test_mode = test_mode
        self.has_ignored_labels = dataset_config['has_ignored_labels']
        
        self.test_probs = [np.zeros(shape=[l.shape[0], dataset.num_classes], dtype=np.float32)
                           for l in dataset.input_labels[test_mode]]

    def test(self, dataset, num_vote=10):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Number of points per class in test set
        val_proportions = np.zeros(dataset.num_classes, dtype=np.float32)

        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels]) # Count how many points in each class
                i += 1
  
        step_id = 0
        epoch_id = 0
        last_min = -0.5        

        while last_min < num_vote:
            stat_dict = {}
            net.eval()
            iou_calc = IoUCalculator(cfg)    

            for batch_idx, batch_data in enumerate(test_dataloader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(len(batch_data[key])):
                            batch_data[key][i] = batch_data[key][i].to(device)
                    else:
                        batch_data[key] = batch_data[key].to(device)

                # Forward pass
                with torch.no_grad():
                    end_points = net(batch_data)

                loss, end_points = compute_loss(end_points, cfg, device)

                # For DALES, use unfiltered logits for proper reshaping
                if self.has_ignored_labels:
                    stacked_probs = end_points['logits'].transpose(1, 2).reshape(-1, cfg.num_classes)
                    stacked_labels = end_points['labels'].reshape(-1)
                else:
                    stacked_probs = end_points['valid_logits']
                    stacked_labels = end_points['valid_labels']
                
                point_idx = end_points['input_inds'].cpu().numpy()
                cloud_idx = end_points['cloud_inds'].cpu().numpy()

                # Reshape and apply softmax
                batchsize = end_points['logits'].shape[0]
                stacked_probs = torch.reshape(stacked_probs, [batchsize, cfg.num_points, cfg.num_classes])
                stacked_probs = F.softmax(stacked_probs, dim=2).cpu().numpy()
                stacked_labels = stacked_labels.cpu().numpy()

                for j in range(np.shape(stacked_probs)[0]):     # Process each batch
                    probs = stacked_probs[j, :, :]      # Prediction results (scores) for current batch
                    p_idx = point_idx[j, :]             # Point indices for current batch
                    c_i = cloud_idx[j][0]               # Which scene current batch belongs to
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs # Voting operation, average multiple inference results for more stable/smooth results
                step_id += 1

            new_min = np.min(test_dataset.min_possibility[self.test_mode])
            log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                log_string('\nConfusion on sub clouds')                                 # Results below are on grid-sampled sub-clouds
                confusion_list = []

                num_val = len(dataset.input_labels[self.test_mode])

                for i_test in range(num_val):
                    probs = self.test_probs[i_test]
                    
                    if self.has_ignored_labels:
                        # DALES: Network outputs indices 0-7 for classes 1-8 (excluding ignored class 0)
                        pred_indices = np.argmax(probs, axis=1)
                        preds = pred_indices + 1  # Map network output indices to actual labels: 0->1, 1->2, ..., 7->8
                        labels = dataset.input_labels[self.test_mode][i_test]
                        
                        # Filter out ignored labels (label 0)
                        valid_mask = labels != 0
                        labels_valid = labels[valid_mask]
                        preds_valid = preds[valid_mask]
                        
                        # Compute confusion matrix only on valid points (labels 1-8)
                        confusion_list += [confusion_matrix(labels_valid, preds_valid, labels=dataset.label_values[1:])]
                    else:
                        # S3DIS: No ignored labels
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                        labels = dataset.input_labels[self.test_mode][i_test]
                        confusion_list += [confusion_matrix(labels, preds, labels=dataset.label_values)]

                # Aggregate confusions
                C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

                # Rescale with correct proportions
                C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                # Compute IoUs
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:6.3f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:6.3f} '.format(100 * IoU)
                log_string(s + '\n')

                if last_min >= num_vote - 1:

                    # Project predictions to original point clouds
                    log_string('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    proj_probs_list = []

                    for i_val in range(num_val):
                        # Reproject probs back to the evaluations points
                        proj_idx = dataset.val_proj[i_val]                  # Get original point indices for scene i_val
                        probs = self.test_probs[i_val][proj_idx, :]         # This indexing is interesting, related to val_proj generation. This step completes projection from sampled to original point cloud (prediction)
                        proj_probs_list += [probs]                          # Save original point cloud predictions to this list

                    # Show vote results on full clouds
                    log_string('Confusion on full clouds')
                    confusion_list = []
                    
                    for i_test in range(num_val):
                        if self.has_ignored_labels:
                            # DALES: Network outputs indices 0-7 for classes 1-8 (excluding ignored class 0)
                            pred_indices = np.argmax(proj_probs_list[i_test], axis=1)
                            preds = (pred_indices + 1).astype(np.uint8)  # Map 0->1, 1->2, ..., 7->8
                            
                            labels = dataset.val_labels[i_test]
                            
                            # Filter out ignored labels (label 0) for accuracy calculation
                            valid_mask = labels != 0
                            labels_valid = labels[valid_mask]
                            preds_valid = preds[valid_mask]
                            acc = np.sum(preds_valid == labels_valid) / len(labels_valid)
                            log_string(f'{dataset.input_names[self.test_mode][i_test]} Acc: {acc:.4f}')
                            
                            # Compute confusion matrix only on valid points (labels 1-8)
                            confusion_list += [confusion_matrix(labels_valid, preds_valid, labels=dataset.label_values[1:])]
                        else:
                            # S3DIS: No ignored labels
                            preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)
                            labels = dataset.val_labels[i_test]
                            acc = np.sum(preds == labels) / len(labels)
                            log_string(dataset.input_names[self.test_mode][i_test] + ' Acc:' + str(acc))
                            
                            confusion_list += [confusion_matrix(labels, preds, labels=dataset.label_values)]
                        
                        # Save predictions
                        name = dataset.input_names[self.test_mode][i_test] + '.ply'
                        write_ply(os.path.join(LOG_DIR, preds_folder, name), [preds, labels], ['pred', 'label'])

                    # Aggregate confusions
                    C = np.sum(np.stack(confusion_list), axis=0)

                    # Compute IoU
                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    
                    # Compute precision, recall, F1, accuracy
                    PRE, REC, F1, _, ACC = DP.metrics(C.copy())  # Use copy to avoid modifying C
                    m_PRE = np.mean(PRE)
                    m_REC = np.mean(REC)
                    m_F1 = np.mean(F1)
                    
                    # Get class names
                    if self.has_ignored_labels:
                        class_names = [dataset.label_to_names[v] for v in dataset.label_values[1:]]
                    else:
                        class_names = [dataset.label_to_names[v] for v in dataset.label_values]
                    
                    # Log class names header
                    log_string('\n' + '-' * 80)
                    s = '{:>10} | '.format('mean')
                    for name in class_names:
                        s += '{:>12} '.format(name[:12])  # Truncate long names
                    log_string(s)
                    log_string('-' * len(s))
                    
                    # Log IoU
                    s = '{:>9.3f} | '.format(100 * m_IoU)
                    for iou in IoUs:
                        s += '{:>11.3f} '.format(100 * iou)
                    log_string('IoU:      ' + s)
                    
                    # Log Precision
                    s = '{:>9.3f} | '.format(100 * m_PRE)
                    for pre in PRE:
                        s += '{:>11.3f} '.format(100 * pre)
                    log_string('Precision:' + s)
                    
                    # Log Recall
                    s = '{:>9.3f} | '.format(100 * m_REC)
                    for rec in REC:
                        s += '{:>11.3f} '.format(100 * rec)
                    log_string('Recall:   ' + s)
                    
                    # Log F1
                    s = '{:>9.3f} | '.format(100 * m_F1)
                    for f1 in F1:
                        s += '{:>11.3f} '.format(100 * f1)
                    log_string('F1:       ' + s)
                    
                    log_string('-' * 80)
                    log_string(f'Overall Accuracy: {100*ACC:.3f}')
                    
                    print('Finished\n')
                    return

            epoch_id += 1
            step_id = 0
            continue
        
        return


if __name__ == '__main__':
    log_string('**** TESTING STARTED ****')
    log_string(f'Dataset: {FLAGS.dataset}')
    if FLAGS.dataset == 'S3DIS':
        log_string(f'Test area: {FLAGS.test_area}')
    else:
        log_string(f'Mode: {test_mode}')
    log_string(f'Number of test samples: {len(test_dataset)}')
    log_string(f'Number of classes: {cfg.num_classes}')
    
    start_time = time.time()

    try:
        test_model = ModelTester(dataset, test_mode)
        test_model.test(dataset, num_vote=FLAGS.num_votes)
    except Exception as e:
        log_string(f'Error during testing: {e}')
        LOG_FOUT.close()
    
    end_time = time.time()
    duration = end_time - start_time
    
    log_string(f'\nTotal testing time: {duration:.2f}s ({duration/60:.2f}min)')
    
    LOG_FOUT.close()