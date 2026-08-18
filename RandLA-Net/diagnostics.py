# ----------------------------------------------------------------------------------------------------------------------
#
#      Diagnostics script for RandLA-Net models
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import argparse
import os

# Limit thread count per worker to prevent thread exhaustion across many sequential processes
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import gc
import sys
import time
import numpy as np
import torch
import random
import csv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
from helper_tool import ConfigDALES, ConfigS3DIS, ConfigSTPLS3D

# Dataset configuration mapping
DATASET_CONFIG = {
    'S3DIS': {
        'config_class': ConfigS3DIS,
        'module': 'datasets.S3DIS',
        'cls_num_list': [37334028, 32206900, 53133563, 4719832, 4145093, 4127868, 10681455, 6318085, 7930065, 949299, 9188737, 2457821, 21826246],
        'has_test_area': True,
    },
    'DALES': {
        'config_class': ConfigDALES,
        'module': 'datasets.DALES',
        'cls_num_list': [171803872, 120576463, 2567061, 744729, 777716, 1482534, 267847, 56648714],
        'has_test_area': False,
    },
    'STPLS3D': {
        'config_class': ConfigSTPLS3D,
        'module': 'datasets.STPLS3D',
        'cls_num_list': [847895239, 576172997, 703621125, 33220987, 8391386, 19030897],
        'has_test_area': False,
    },
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Use True when input is fixed


def print_model_info(net, cfg, dataset_name):
    """
    Print information about the loaded RandLA-Net model.
    
    Args:
        net: The loaded network
        cfg: The configuration object
        dataset_name: Name of the dataset
    """
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    print(f"Dataset: {dataset_name}")
    print(f"Number of classes: {cfg.num_classes}")
    print(f"Number of layers: {cfg.num_layers}")
    print(f"Feature dimensions (d_out): {cfg.d_out}")
    print(f"Input points: {cfg.num_points}")
    print(f"Subsampling ratios: {cfg.sub_sampling_ratio}")
    
    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(f"Model device: {next(net.parameters()).device}")
    
    # Loss function and weight information
    print("\n" + "-"*30)
    print("LOSS CONFIGURATION")
    print("-"*30)
    
    loss_fn = getattr(cfg, 'loss_fn', None)
    if loss_fn is not None:
        print(f"Loss function: {type(loss_fn).__name__}")
    else:
        print("Loss function: CrossEntropyLoss (default)")
    
    # Class weights
    if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
        print(f"Class weights: {cfg.class_weights}")
    else:
        print("Class weights: None (uniform weights)")
    
    print("="*50)


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def get_random_weights(weights):
    """
    Produce a random direction that is a list of random Gaussian tensors
    with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size(), device=w.device) for w in weights]


def normalize_direction(direction, weights, norm='filter'):
    """
    Rescale the direction so that it has similar norm as their corresponding
    model in different levels.

    Args:
      direction: a variables of the random direction for one layer
      weights: a variable of the original model for one layer
      norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
    The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)




def compute_validation_iou(net, val_loader, cfg, device):
    """
    Compute validation IoU for RandLA-Net point cloud segmentation.
    
    Args:
        net: The trained neural network
        val_loader: DataLoader for validation set
        cfg: Configuration object
        device: Device to run computations on
        
    Returns:
        tuple: (mean_iou, per_class_ious list)
    """
    net.eval()
    iou_calc = IoUCalculator(cfg)
    
    with torch.no_grad():
        for batch_data in val_loader:
            # Move batch to device
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)
            
            # Forward pass
            end_points = net(batch_data)
            loss, end_points = compute_loss(end_points, cfg, device)
            acc, end_points = compute_acc(end_points)
            iou_calc.add_data(end_points)
    
    mean_iou, iou_list = iou_calc.compute_iou()
    
    return float(mean_iou), iou_list


def calculate_flatness_metric(net, train_loader, val_loader, cfg, device, desired_percent=0.1, K=20):
    """
    Calculate flatness metric by evaluating loss and validation IoU perturbations around optimal weights.
    
    Args:
        net: The trained neural network
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        cfg: Configuration object
        device: Device to run computations on
        desired_percent: Desired perturbation as percentage of weight norm (default: 0.1)
        K: Number of random directions to sample (default: 20)
        
    Returns:
        dict: Dictionary containing flatness metrics
    """
    print(f"\nCalculating flatness metric with desired perturbation={desired_percent}%, K={K}...")
    
    net.eval()
    
    # Store original weights (maintaining tensor shapes)
    original_weights = []
    total_params = 0
    
    for param in net.parameters():
        if param.requires_grad and len(param.size()) > 1:
            original_weights.append(param.data.clone())
            total_params += param.numel()
      
    print(f"Total parameters: {total_params:,}")
    loss_fn = getattr(cfg, 'loss_fn', None)
    if loss_fn is not None:
        print(f"Using loss function: {type(loss_fn).__name__}")
    else:
        print("Using default CrossEntropyLoss")

    # -------------------------------------- Function definition -------------------------------------- #
    def evaluate_loss_grad_norm(model, calc_grad_norm=False):
        """
        Evaluate loss on both train and val datasets, optionally computing gradient norm on train set.
        
        Returns:
            tuple: (train_loss, val_loss, grad_norm) where grad_norm is None if calc_grad_norm=False
        """
        model.eval()
        
        # Compute training loss (with or without gradients)
        train_total_loss = 0.0
        train_num_samples = 0
        
        model.zero_grad()
        
        if calc_grad_norm:
            for batch_data in train_loader:
                # Move batch to device
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(len(batch_data[key])):
                            batch_data[key][i] = batch_data[key][i].to(device)
                    else:
                        batch_data[key] = batch_data[key].to(device)

                end_points = model(batch_data)
                loss, end_points = compute_loss(end_points, cfg, device)
                
                num_valid_points = end_points['valid_labels'].numel()
                train_total_loss += loss.item() * num_valid_points
                train_num_samples += num_valid_points
                
                loss.backward()
            
            # Calculate gradient norm
            grad_norm = 0.0
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm += param.grad.data.norm(2) ** 2
            grad_norm = torch.sqrt(grad_norm).item()

        else:
            with torch.no_grad():
                for batch_data in train_loader:
                    # Move batch to device
                    for key in batch_data:
                        if type(batch_data[key]) is list:
                            for i in range(len(batch_data[key])):
                                batch_data[key][i] = batch_data[key][i].to(device)
                        else:
                            batch_data[key] = batch_data[key].to(device)

                    end_points = model(batch_data)
                    loss, end_points = compute_loss(end_points, cfg, device)
                    
                    num_valid_points = end_points['valid_labels'].numel()
                    train_total_loss += loss.item() * num_valid_points
                    train_num_samples += num_valid_points

            grad_norm = None
        
        train_loss = train_total_loss / train_num_samples if train_num_samples > 0 else 0.0
        
        # Compute validation loss (always without gradients)
        val_total_loss = 0.0
        val_num_samples = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Move batch to device
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(len(batch_data[key])):
                            batch_data[key][i] = batch_data[key][i].to(device)
                    else:
                        batch_data[key] = batch_data[key].to(device)

                end_points = model(batch_data)
                loss, end_points = compute_loss(end_points, cfg, device)
                
                num_valid_points = end_points['valid_labels'].numel()
                val_total_loss += loss.item() * num_valid_points
                val_num_samples += num_valid_points
        
        val_loss = val_total_loss / val_num_samples if val_num_samples > 0 else 0.0
        
        return train_loss, val_loss, grad_norm
    # -------------------------------------------------------------------------------------------------- #

    print("Evaluating original losses and gradient norm L(θ*), ||∇L(θ*)||...")
    original_train_loss, original_val_loss, original_grad_norm = evaluate_loss_grad_norm(net, calc_grad_norm=True)
    print(f"Original train loss L_train(θ*): {original_train_loss:.6f}")
    print(f"Original val loss L_val(θ*): {original_val_loss:.6f}")
    print(f"Original gradient norm ||∇L(θ*)||: {original_grad_norm:.8f}")
    
    print("Computing original validation IoU...")
    original_mean_iou, original_per_class_ious = compute_validation_iou(net, val_loader, cfg, device)
    print(f"Original mean IoU: {original_mean_iou:.4f}")
    print(f"Original per-class IoUs: {[f'{iou:.4f}' for iou in original_per_class_ious]}")
    
    print(f"\nSampling {K} random directions and evaluating perturbed losses...")
    perturbed_train_losses = []
    perturbed_val_losses = []
    perturbed_grad_norms = []
    perturbed_mean_ious = []
    perturbed_per_class_ious = []
    all_sampled_directions = []

    for k in range(K):
        # Generate random direction (NeurIPS 2017 approach)
        directions = get_random_weights(original_weights)
        
        # Normalize directions using filter-wise normalization
        normalize_directions_for_weights(directions, original_weights, norm='filter', ignore='biasbn')
        
        all_sampled_directions.append(directions)
        
        # # Compute relative perturbation: alpha*d / w, i.e. (w + alpha*d - w) / w
        # perturbation_norm_unit = 0.0
        # weight_norm = 0.0
        # for w, d in zip(original_weights, directions):
        #     perturbation_norm_unit += (1.0 * d).norm().item() ** 2   # alpha = 1
        #     weight_norm += w.norm().item() ** 2
        # perturbation_norm_unit = np.sqrt(perturbation_norm_unit)
        # weight_norm = np.sqrt(weight_norm)
        # rho_unit = perturbation_norm_unit / (weight_norm + 1e-12)

        # target_rho = desired_percent / 100.0

        # # compute alpha that yields target_rho
        # if rho_unit == 0:
        #     alpha = 0.0
        # else:
        #     alpha = target_rho / rho_unit

        alpha = desired_percent / 100.0  # Direct scaling approach
        
        # Set perturbed weights: w* + alpha * direction
        trainable_params = [p for p in net.parameters() if p.requires_grad and len(p.size()) > 1]
        for (param, w, d) in zip(trainable_params, original_weights, directions):
            param.data.copy_(w + alpha * d)

        perturbed_train_loss, perturbed_val_loss, perturbed_grad_norm = evaluate_loss_grad_norm(net, calc_grad_norm=True)
        perturbed_train_losses.append(perturbed_train_loss)
        perturbed_val_losses.append(perturbed_val_loss)
        perturbed_grad_norms.append(perturbed_grad_norm)
        
        perturbed_mean_IoU, perturbed_per_class_IoU = compute_validation_iou(net, val_loader, cfg, device)
        perturbed_mean_ious.append(perturbed_mean_IoU)
        perturbed_per_class_ious.append(perturbed_per_class_IoU)

        print(f"Direction {k+1}/{K}: ρ = {alpha:.6f}, L_train = {perturbed_train_loss:.6f}, L_val = {perturbed_val_loss:.6f}, mIoU = {perturbed_mean_IoU:.4f}")

    # Restore original parameters
    trainable_params = [p for p in net.parameters() if p.requires_grad and len(p.size()) > 1]
    for (param, w) in zip(trainable_params, original_weights):
        param.data.copy_(w)
    

    train_loss_differences = np.array([loss - original_train_loss for loss in perturbed_train_losses])
    val_loss_differences = np.array([loss - original_val_loss for loss in perturbed_val_losses])
    
    iou_differences = np.array([original_mean_iou - iou for iou in perturbed_mean_ious])

    # Calculate orthogonality of sampled directions
    all_directions_flat = []
    for directions in all_sampled_directions:
        flat_dir = torch.cat([d.flatten() for d in directions])
        all_directions_flat.append(flat_dir)
    
    if len(all_directions_flat) > 0:
        direction_matrix = torch.stack(all_directions_flat)  # [K, total_params]

        # Normalize each direction to unit length
        direction_norms = torch.norm(direction_matrix, dim=1, keepdim=True)
        direction_matrix_normalized = direction_matrix / (direction_norms + 1e-10)

        # Compute Gram matrix with normalized directions
        gram_matrix = torch.mm(direction_matrix_normalized, direction_matrix_normalized.t())  # [K, K]
        
        # Off-diagonal elements should be close to 0 for orthogonal directions
        off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=gram_matrix.device)
        mean_dot_product = gram_matrix[off_diag_mask].abs().mean().item()
    else:
        mean_dot_product = 0.0
    
    print(f"\nFlatness Metric Results:")
    print(f"Original train loss L_train(θ*): {original_train_loss:.6f}")
    print(f"Original val loss L_val(θ*): {original_val_loss:.6f}")
    print(f"Original gradient norm ||∇L(θ*)||: {original_grad_norm:.8f}")
    print(f"Original mean IoU: {original_mean_iou:.4f}")
    print(f"Mean orthogonality: {mean_dot_product:.6f}")
    
    return {
        'original_train_loss': original_train_loss,
        'original_val_loss': original_val_loss,
        'original_grad_norm': original_grad_norm,
        'original_mean_iou': original_mean_iou,
        'original_per_class_ious': original_per_class_ious,
        'perturbed_train_losses': perturbed_train_losses,
        'perturbed_val_losses': perturbed_val_losses,
        'perturbed_grad_norms': perturbed_grad_norms,
        'perturbed_mean_ious': perturbed_mean_ious,
        'perturbed_per_class_ious': perturbed_per_class_ious,
        'train_loss_differences': train_loss_differences.tolist(),
        'val_loss_differences': val_loss_differences.tolist(),
        'iou_differences': iou_differences.tolist(),
        'mean_orthogonality': mean_dot_product,
        'desired_percent': desired_percent,
        'K': K
    }


def setup_data_loaders(dataset_name, cfg, test_area=None):
    """
    Setup data loaders for RandLA-Net datasets.
    
    Args:
        dataset_name: Name of the dataset ('DALES', 'S3DIS', 'STPLS3D', etc.)
        cfg: Configuration object
        test_area: Test area index for S3DIS dataset
        
    Returns:
        tuple: (training_loader, validation_loader)
    """
    from torch.utils.data import DataLoader
    
    print("\nSetting up data loaders...")
    
    if dataset_name == 'DALES':
        from datasets.DALES import DALES, DALESSampler
        
        dataset_train = DALES('training')
        dataset_val = DALES('validation')
        training_dataset = DALESSampler(dataset_train, 'training')
        validation_dataset = DALESSampler(dataset_val, 'validation')
        
    elif dataset_name == 'S3DIS':
        from datasets.S3DIS import S3DIS, S3DISSampler
        
        if test_area is None:
            test_area = 5  # Default to Area 5
        dataset = S3DIS(test_area)
        training_dataset = S3DISSampler(dataset, 'training')
        validation_dataset = S3DISSampler(dataset, 'validation')
        
    elif dataset_name == 'STPLS3D':
        from datasets.STPLS3D import STPLS3D, STPLS3DSampler
        
        dataset_train = STPLS3D('training')
        dataset_val = STPLS3D('validation')
        training_dataset = STPLS3DSampler(dataset_train, 'training')
        validation_dataset = STPLS3DSampler(dataset_val, 'validation')
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    training_loader = DataLoader(
        training_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=training_dataset.collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        collate_fn=validation_dataset.collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Training steps per epoch: {len(training_loader)}")
    print(f"Validation steps per epoch: {len(validation_loader)}")
    print(f"Number of classes: {cfg.num_classes}")
        
    return training_loader, validation_loader


def main():
    parser = argparse.ArgumentParser(description="Load and diagnose a trained RandLA-Net model")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the checkpoint file (.tar)")
    parser.add_argument('--dataset', type=str, default='DALES', choices=['DALES', 'S3DIS', 'STPLS3D'],
                        help="Dataset name")
    parser.add_argument('--test_area', type=int, default=5, help="Test area for S3DIS dataset (default: 5)")
    parser.add_argument('--gpu', type=str, default='0', help="GPU ID to use (default: 0)")
    parser.add_argument('--desired_percent', type=str, default='0.1', help="Comma-separated list of perturbation percentages (e.g., '0.01,0.1,1.0,10.0,20.0,30.0')")
    parser.add_argument('--K', type=int, default=20, help="Number of random directions to sample (default: 20)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducible results (default: 42)")
    parser.add_argument('--compute_flatness', action='store_true', help="Compute flatness metric by sampling random directions")
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setup_seed(args.seed)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse comma-separated desired_percent into list of floats
    desired_percents = [float(p.strip()) for p in args.desired_percent.split(',')]

    try:
        # Check checkpoint exists
        if not os.path.exists(args.checkpoint_path):
            raise ValueError(f"Checkpoint file not found: {args.checkpoint_path}")
        
        print(f"Loading checkpoint: {args.checkpoint_path}")
        
        # Get dataset configuration
        if args.dataset not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        cfg = DATASET_CONFIG[args.dataset]['config_class']
        cfg.name = args.dataset
        
        # Set ignored label indices for the dataset
        if args.dataset == 'DALES':
            cfg.ignored_label_inds = [0]  # Unknown class
        elif args.dataset == 'S3DIS':
            cfg.ignored_label_inds = []  # No ignored labels in S3DIS
        elif args.dataset == 'STPLS3D':
            cfg.ignored_label_inds = []  # No ignored labels in STPLS3D
        
        # Output directory (same as checkpoint directory)
        output_dir = os.path.dirname(args.checkpoint_path)
        
        print(f"Dataset: {args.dataset}")
        print(f"Number of classes: {cfg.num_classes}")

        # Setup data loaders
        training_loader, validation_loader = setup_data_loaders(
            args.dataset, cfg, 
            test_area=args.test_area if args.dataset == 'S3DIS' else None
        )

        print("Initializing network...")
        t1 = time.time()
        net = Network(cfg)
        net.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        
        print(f"Model loaded in {time.time() - t1:.1f}s")
        
        print_model_info(net, cfg, args.dataset)
        
        print(f"\nModel successfully loaded from epoch {epoch}")
        print("Model is ready for diagnostics and analysis.")
        
        # -------------------------------------- Flatness Metric Calculation ------------------------------------- #
        if args.compute_flatness:
            print("\n" + "="*50)
            print("FLATNESS METRIC ANALYSIS")
            print("="*50)
            print(f"Percentages to evaluate: {desired_percents}")
            
            for desired_percent in desired_percents:
                # Skip if already computed
                summary_filename = f'flatness_summary_percent_{desired_percent}_K_{args.K}_seed_{args.seed}.csv'
                summary_file = os.path.join(output_dir, summary_filename)
                if os.path.exists(summary_file):
                    print(f"\n  Skipping {desired_percent}% (already computed: {summary_filename})")
                    continue
                
                print(f"\nCalculating flatness for desired perturbation = {desired_percent}%")
                
                start_time = time.time()
                flatness_results = calculate_flatness_metric(
                    net=net,
                    train_loader=training_loader,
                    val_loader=validation_loader,
                    cfg=cfg,
                    device=device,
                    desired_percent=desired_percent,
                    K=args.K
                )
                end_time = time.time()
                execution_time = end_time - start_time
                
                print(f"Flatness calculation completed in {execution_time:.2f} seconds")
                
                # File 1: Summary statistics
                num_classes = len(flatness_results['original_per_class_ious'])
                
                # Build summary data as list of [metric_name, value] pairs
                summary_data = [
                    ['metric', 'value'],
                    ['desired_percent', desired_percent],
                    ['K', args.K],
                    ['original_train_loss', flatness_results['original_train_loss']],
                    ['original_val_loss', flatness_results['original_val_loss']],
                    ['original_grad_norm', flatness_results['original_grad_norm']],
                    ['original_mean_iou', flatness_results['original_mean_iou']],
                    ['mean_orthogonality', flatness_results['mean_orthogonality']],
                    ['execution_time_seconds', execution_time]
                ]
                
                # Add original per-class IoUs
                for class_idx in range(num_classes):
                    summary_data.append([f'original_class_{class_idx}_iou', flatness_results['original_per_class_ious'][class_idx]])
                
                with open(summary_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(summary_data)
                
                print(f"Summary statistics saved to: {summary_file}")
                
                # File 2: Per-sample results
                samples_filename = f'flatness_samples_percent_{desired_percent}_K_{args.K}_seed_{args.seed}.csv'
                samples_file = os.path.join(output_dir, samples_filename)
                
                # Define columns for samples file
                sample_columns = ['sample_id', 'perturbed_train_loss', 'perturbed_val_loss', 'perturbed_grad_norm', 'perturbed_mean_iou']
                for class_idx in range(num_classes):
                    sample_columns.append(f'perturbed_class_{class_idx}_iou')
                
                with open(samples_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sample_columns)
                    writer.writeheader()
                    
                    for k_idx in range(args.K):
                        sample_row = {
                            'sample_id': k_idx + 1,
                            'perturbed_train_loss': flatness_results['perturbed_train_losses'][k_idx],
                            'perturbed_val_loss': flatness_results['perturbed_val_losses'][k_idx],
                            'perturbed_grad_norm': flatness_results['perturbed_grad_norms'][k_idx],
                            'perturbed_mean_iou': flatness_results['perturbed_mean_ious'][k_idx]
                        }
                        
                        # Add per-class IoUs for this sample
                        for class_idx in range(num_classes):
                            sample_row[f'perturbed_class_{class_idx}_iou'] = flatness_results['perturbed_per_class_ious'][k_idx][class_idx]
                        
                        writer.writerow(sample_row)
                
                print(f"Per-sample results saved to: {samples_file}")
        
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Explicit cleanup to prevent CUDA resource leaks between sequential bash script calls
        if 'training_loader' in locals():
            del training_loader
        if 'validation_loader' in locals():
            del validation_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
