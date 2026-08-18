"""
Unified training script for RandLA-Net.
"""

import os

import argparse
import warnings
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime

from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
from losses import BalancedSoftmax, FocalLoss, LADJLoss, LDAMLoss, SeesawLoss


# ================================================================================
# Dataset-specific configurations
# ================================================================================

DATASET_CONFIG = {
    'S3DIS': {
        'config_class': 'ConfigS3DIS',
        'dataset_class': 'S3DIS',
        'sampler_class': 'S3DISSampler',
        'module': 'datasets.S3DIS',
        'log_dir': 'results/S3DIS',
        # Classes: ceiling, floor, wall, beam, column, window, door, table, chair, sofa, bookcase, board, clutter
        'cls_num_list': [37334028, 32206900, 53133563, 4719832, 4145093, 4127868, 10681455, 6318085, 7930065, 949299, 9188737, 2457821, 21826246],
        'has_test_area': True,
        'num_workers': 0,
    },
    'DALES': {
        'config_class': 'ConfigDALES',
        'dataset_class': 'DALES',
        'sampler_class': 'DALESSampler',
        'module': 'datasets.DALES',
        'log_dir': 'results/DALES',
        # Classes: Ground, Vegetation, Cars, Trucks, Power lines, Fences, Poles, Buildings (excluding Unknown)
        'cls_num_list': [171803872, 120576463, 2567061, 744729, 777716, 1482534, 267847, 56648714],
        'has_test_area': False,
        'num_workers': 0,
    },
    'STPLS3D': {
        'config_class': 'ConfigSTPLS3D',
        'dataset_class': 'STPLS3D',
        'sampler_class': 'STPLS3DSampler',
        'module': 'datasets.STPLS3D',
        'log_dir': 'results/STPLS3D',
        'cls_num_list': [847895239, 576172997, 703621125, 33220987, 8391386, 19030897],
        'has_test_area': False,
        'num_workers': 0,
    },
}


def get_dataset_components(dataset_name):
    """
    Dynamically import and return dataset-specific components.
    
    Returns:
        cfg: Configuration class instance
        dataset_class: Dataset class
        sampler_class: Sampler class
    """
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


def create_loss_function(loss_name, cfg, cls_num_list):
    """Create the loss function based on the specified name."""

    if loss_name == 'ce':
        return None  # Use default CrossEntropyLoss in compute_loss
    
    elif loss_name == 'balanced_softmax':
        if cls_num_list is None:
            raise ValueError("balanced_softmax requires cls_num_list")
        return BalancedSoftmax(cls_num_list)
    
    elif loss_name == 'focal':
        return FocalLoss(num_class=cfg.num_classes)
    
    elif loss_name == 'ladj':
        if cls_num_list is None:
            raise ValueError("ladj requires cls_num_list")
        return LADJLoss(cls_num_list)
    
    elif loss_name == 'ldam':
        if cls_num_list is None:
            raise ValueError("ldam requires cls_num_list")
        return LDAMLoss(cls_num_list)
    
    elif loss_name == 'seesaw':
        return SeesawLoss(num_classes=cfg.num_classes)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def create_dataloaders(dataset_name, dataset_class, sampler_class, cfg, test_area=None, num_workers=0):

    if dataset_name == 'S3DIS':
        dataset = dataset_class(test_area)
        training_dataset = sampler_class(dataset, 'training')
        validation_dataset = sampler_class(dataset, 'validation')

    elif dataset_name == 'DALES':
        dataset_train = dataset_class('training')
        dataset_val = dataset_class('validation')
        training_dataset = sampler_class(dataset_train, 'training')
        validation_dataset = sampler_class(dataset_val, 'validation')

    elif dataset_name == 'STPLS3D':
        dataset_train = dataset_class('training')
        dataset_val = dataset_class('validation')
        training_dataset = sampler_class(dataset_train, 'training')
        validation_dataset = sampler_class(dataset_val, 'validation')

    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet")
    
    training_dataloader = DataLoader(
        training_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=training_dataset.collate_fn
    )
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=cfg.val_batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=validation_dataset.collate_fn
    )
    
    return training_dataloader, validation_dataloader


class Trainer:
    """
    Unified trainer class for RandLA-Net.
    """
    
    def __init__(self, args, cfg, training_dataloader, validation_dataloader, device, log_dir):
        self.args = args
        self.cfg = cfg
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.log_dir = log_dir
        self.epoch_cnt = 0
        
        # Setup logging
        self._setup_logging()
        
        # Create network
        self.net = Network(cfg)
        self.net.to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)  # Not used, LR adjusted manually
        
        # Load checkpoint if provided
        self.start_epoch = 0
        if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
            self._load_checkpoint(args.checkpoint_path)
    
    def _setup_logging(self):
        """Setup logging directory and file."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create log filename based on dataset
        if self.args.dataset == 'S3DIS':
            log_file_name = f'log_train_Area_{self.args.test_area:d}.txt'
        else:
            log_file_name = 'log_train.txt'
        
        self.log_fout = open(os.path.join(self.log_dir, log_file_name), 'a')
    
    def log_string(self, out_str):
        """Write to log file and print to console."""
        self.log_fout.write(out_str + '\n')
        self.log_fout.flush()
        print(out_str)
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.log_string(f"-> loaded checkpoint {checkpoint_path} (epoch: {self.start_epoch})")
    
    def _save_checkpoint(self, epoch, loss):
        """Save model checkpoint."""
        save_dict = {
            'epoch': epoch + 1,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except:
            save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, os.path.join(self.log_dir, 'checkpoint.tar'))
    
    def adjust_learning_rate(self, epoch):
        """Adjust learning rate based on epoch."""
        lr = self.optimizer.param_groups[0]['lr']
        lr = lr * self.cfg.lr_decays[epoch]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_one_epoch(self):
        
        stat_dict = {}
        self.adjust_learning_rate(self.epoch_cnt)
        self.net.train()
        
        iou_calc = IoUCalculator(self.cfg)
        epoch_start_time = time.time()
        
        for batch_idx, batch_data in enumerate(self.training_dataloader):
            t_start = time.time()
            
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].to(self.device)
                else:
                    batch_data[key] = batch_data[key].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            end_points = self.net(batch_data)
            
            loss, end_points = compute_loss(end_points, self.cfg, self.device)
            loss.backward()
            self.optimizer.step()
            
            acc, end_points = compute_acc(end_points)
            iou_calc.add_data(end_points)
            
            # Accumulate statistics
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'iou' in key:
                    if key not in stat_dict:
                        stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()
            
            batch_interval = 50
            if (batch_idx + 1) % batch_interval == 0:
                t_end = time.time()
                self.log_string(
                    'Step %03d Loss %.3f Acc %.2f lr %.5f --- %.2f ms/batch' % (
                        batch_idx + 1, 
                        stat_dict['loss'] / batch_interval, 
                        stat_dict['acc'] / batch_interval, 
                        self.optimizer.param_groups[0]['lr'], 
                        1000 * (t_end - t_start)
                    )
                )
                stat_dict['loss'], stat_dict['acc'] = 0, 0
        
        mean_iou, iou_list = iou_calc.compute_iou()
        self.log_string('mean IoU:{:.3f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:6.3f} '.format(100 * iou_tmp)
        self.log_string(s)
        epoch_time = time.time() - epoch_start_time
        self.log_string('Training epoch time: {:.2f} seconds ({:.2f} minutes)'.format(epoch_time, epoch_time / 60))
    

    def evaluate_one_epoch(self):
    
        stat_dict = {}
        self.net.eval()
        iou_calc = IoUCalculator(self.cfg)
        eval_start_time = time.time()
        
        for batch_idx, batch_data in enumerate(self.validation_dataloader):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].to(self.device)
                else:
                    batch_data[key] = batch_data[key].to(self.device)
            
            # Forward pass
            with torch.no_grad():
                end_points = self.net(batch_data)
            
            loss, end_points = compute_loss(end_points, self.cfg, self.device)
            acc, end_points = compute_acc(end_points)
            iou_calc.add_data(end_points)
            
            # Accumulate statistics
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'iou' in key:   # No iou item here, iou is calculated below
                    if key not in stat_dict:
                        stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()
        
        for key in sorted(stat_dict.keys()):
            self.log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))
        
        mean_iou, iou_list = iou_calc.compute_iou()
        self.log_string('mean IoU:{:.3f}%'.format(mean_iou * 100))
        self.log_string('-' * 86)
        s = f'{mean_iou * 100:.3f} | '
        for iou_tmp in iou_list:
            s += '{:6.3f} '.format(100 * iou_tmp)
        self.log_string(s)
        self.log_string('-' * 86)
        eval_time = time.time() - eval_start_time
        self.log_string('Evaluation time: {:.2f} seconds ({:.2f} minutes)'.format(eval_time, eval_time / 60))
        
        return mean_iou
    
    def train(self):

        loss = 0
        now_miou = 0
        max_miou = 0
        total_training_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.args.max_epoch):
            self.epoch_cnt = epoch
            self.log_string('**** EPOCH %03d ****' % epoch)
            self.log_string(str(datetime.now()))
            
            self.train_one_epoch()
            
            self.log_string('**** EVAL EPOCH %03d START****' % epoch)
            now_miou = self.evaluate_one_epoch()
            
            # Save checkpoint if best
            if now_miou > max_miou:
                self._save_checkpoint(epoch, loss)
                max_miou = now_miou
            
            self.log_string('Best mIoU = {:2.2f}%'.format(max_miou * 100))
            self.log_string('**** EVAL EPOCH %03d END****' % epoch)
            self.log_string('')
        
        total_training_time = time.time() - total_training_start_time
        self.log_string('=' * 80)
        self.log_string('Total training time: {:.2f} minutes ({:.2f} hours)'.format(
            total_training_time / 60, total_training_time / 3600))
        self.log_string('=' * 80)
    
    def close(self):
        """Close log file."""
        self.log_fout.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RandLA-Net on various datasets')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='S3DIS',
                        choices=['S3DIS', 'DALES', 'STPLS3D'],
                        help='Dataset to train on [default: S3DIS]')
    
    # Common arguments
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default=None, help='Dump dir to save model checkpoint [default: results/<dataset>]')
    parser.add_argument('--max_epoch', type=int, default=100, help='Epochs to run [default: 100]')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use [default: 0]')
    
    # Training options
    parser.add_argument('--weights', type=str, default='none',
                        choices=['none', 'invf', 'cb', 'invl', 'invp', 'comf'],
                        help='Class weighting scheme [default: none]')
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'balanced_softmax', 'focal', 'ladj', 'ldam', 'seesaw'],
                        help='Loss function to use [default: ce]')
    
    # Dataset-specific arguments
    parser.add_argument('--test_area', type=int, default=5,
                        help='Which area to use for test (S3DIS only), option: 1-6 [default: 5]')
    
    return parser.parse_args()


def main():
    # Disable cuDNN (can cause issues with large data matrices on single GPU)
    torch.backends.cudnn.enabled = False
    
    args = parse_args()
    
    dataset_config = DATASET_CONFIG[args.dataset]
    
    # Set default log directory if not specified
    if args.log_dir is None:
        args.log_dir = dataset_config['log_dir']
    
    # Create log directory with timestamp
    log_dir = os.path.join(
        args.log_dir,
        time.strftime(f'{args.dataset}-%Y-%m-%d_%H-%M-%S', time.gmtime()) + f'_{args.weights}_{args.loss}'
    )
    
    # Get dataset components
    cfg, dataset_class, sampler_class = get_dataset_components(args.dataset)
    
    # Set weighting scheme in config
    cfg.weights = args.weights
    
    # Create loss function
    cls_num_list = dataset_config['cls_num_list']
    loss_fn = create_loss_function(args.loss, cfg, cls_num_list)
    cfg.loss_fn = loss_fn
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
        device = torch.device('cpu')
    
    # Create dataloaders
    training_dataloader, validation_dataloader = create_dataloaders(
        args.dataset, dataset_class, sampler_class, cfg, args.test_area,
        num_workers=dataset_config['num_workers']
    )
    print(f'Training: {len(training_dataloader)}, Validation: {len(validation_dataloader)}')
    
    trainer = Trainer(args, cfg, training_dataloader, validation_dataloader, device, log_dir)
    
    # Log initial info
    trainer.log_string('**** TRAINING STARTED ****')
    trainer.log_string(f'Dataset: {args.dataset}')
    trainer.log_string(f'Number of classes: {cfg.num_classes}')
    trainer.log_string(f'Using class weighting scheme: {args.weights}')
    trainer.log_string(f'Using loss function: {args.loss}')
    if args.loss == 'ce':
        trainer.log_string(f'Class weights: {cfg.class_weights}')
    if args.dataset == 'S3DIS':
        trainer.log_string(f'Test area: {args.test_area}')
    
    trainer.train()
    
    trainer.log_string('**** TRAINING FINISHED ****')
    trainer.close()


if __name__ == '__main__':
    main()
