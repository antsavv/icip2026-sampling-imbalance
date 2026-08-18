#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Diagnostics script for KPConv models
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import argparse
import os

import gc
import sys
import time
import traceback
import numpy as np
import torch
import random
import csv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPFCNN

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Use True when input is fixed


def print_model_info(net, config):
    """
    Print information about the loaded model.
    
    Args:
        net: The loaded network
        config: The configuration object
    """
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    print(f"Dataset: {config.dataset}")
    print(f"Dataset task: {config.dataset_task}")
    print(f"Number of classes: {config.num_classes}")
    print(f"Architecture: {config.architecture}")
    
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
    
    print(f"Loss function: {config.loss_function}")
    
    # Class weights
    if hasattr(config, 'class_w') and config.class_w:
        print(f"Class weights: {config.class_w}")
    else:
        print("Class weights: None (uniform weights)")
    
    # Loss-specific parameters
    if config.loss_function == 'focal_loss':
        if hasattr(config, 'focal_loss_gamma'):
            print(f"Focal loss gamma: {config.focal_loss_gamma}")
        if hasattr(config, 'focal_loss_alpha'):
            print(f"Focal loss alpha: {config.focal_loss_alpha}")
        if hasattr(config, 'focal_loss_mode'):
            print(f"Focal loss mode: {config.focal_loss_mode}")
    
    elif config.loss_function == 'ldam_loss':
        if hasattr(config, 'ldam_loss_max_m'):
            print(f"LDAM loss max margin: {config.ldam_loss_max_m}")
        if hasattr(config, 'ldam_loss_s'):
            print(f"LDAM loss scale: {config.ldam_loss_s}")
        if hasattr(config, 'cls_num_list') and config.cls_num_list:
            print(f"Class number list: {config.cls_num_list}")
    
    elif config.loss_function == 'seesaw_loss':
        if hasattr(config, 'seesaw_loss_p'):
            print(f"Seesaw loss p: {config.seesaw_loss_p}")
        if hasattr(config, 'seesaw_loss_q'):
            print(f"Seesaw loss q: {config.seesaw_loss_q}")
        if hasattr(config, 'seesaw_loss_eps'):
            print(f"Seesaw loss eps: {config.seesaw_loss_eps}")
    
    elif config.loss_function == 'ladj_loss':
        if hasattr(config, 'ladju_tau'):
            print(f"LADJ tau: {config.ladju_tau}")
    
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



def compute_validation_iou(net, val_loader, val_dataset, config):
    """
    Compute validation IoU for cloud segmentation without saving files.
    Extracts essential logic from cloud_segmentation_validation().
    
    Args:
        net: The trained neural network
        val_loader: DataLoader for validation set
        val_dataset: Validation dataset object
        config: Configuration object
        
    Returns:
        dict: Dictionary containing mean IoU and per-class IoUs
    """
    from utils.metrics import IoU_from_confusions, fast_confusion
    
    net.eval()
    device = next(net.parameters()).device
    softmax = torch.nn.Softmax(1)
    
    # Number of classes
    nc_tot = val_dataset.num_classes
    nc_model = config.num_classes
    
    # Initialize validation proportions if needed
    val_proportions = np.zeros(nc_model, dtype=np.float32)
    i = 0
    for label_value in val_dataset.label_values:
        if label_value not in val_dataset.ignored_labels:
            val_proportions[i] = np.sum([np.sum(labels == label_value) for labels in val_dataset.validation_labels])
            i += 1
    
    predictions = []
    targets = []
    
    # Validation loop
    with torch.no_grad():
        for batch in val_loader:
            if 'cuda' in device.type:
                batch.to(device)
            
            # Forward pass
            outputs = net(batch, config)
            
            # Get predictions
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            labels = batch.labels.cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            
            # Process each instance in batch
            i0 = 0
            for b_i, length in enumerate(lengths):
                target = labels[i0:i0 + length]
                probs = stacked_probs[i0:i0 + length]
                
                predictions.append(probs)
                targets.append(target)
                i0 += length
    
    # Compute confusion matrices
    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
    for i, (probs, truth) in enumerate(zip(predictions, targets)):
        # Insert false columns for ignored labels
        for l_ind, label_value in enumerate(val_dataset.label_values):
            if label_value in val_dataset.ignored_labels:
                probs = np.insert(probs, l_ind, 0, axis=1)
        
        # Predicted labels
        preds = val_dataset.label_values[np.argmax(probs, axis=1)]
        
        # Confusion matrix
        Confs[i, :, :] = fast_confusion(truth, preds, val_dataset.label_values).astype(np.int32)
    
    # Sum all confusions
    C = np.sum(Confs, axis=0).astype(np.float32)
    
    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(val_dataset.label_values))):
        if label_value in val_dataset.ignored_labels:
            C = np.delete(C, l_ind, axis=0)
            C = np.delete(C, l_ind, axis=1)
    
    # Balance with real validation proportions
    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
    
    # Compute per-class IoUs
    per_class_ious = IoU_from_confusions(C)
    mean_iou = np.mean(per_class_ious)
    
    return float(mean_iou), per_class_ious.tolist()


def calculate_flatness_metric(net, train_loader, val_loader, val_dataset, config, desired_percent=0.1, K=20):
    """
    Calculate flatness metric by evaluating loss and validation IoU perturbations around optimal weights.
    
    Args:
        net: The trained neural network
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        val_dataset: Validation dataset object
        config: Configuration object containing loss function info
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
    print(f"Using model's built-in loss function: {config.loss_function}")    

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
            for batch in train_loader:
                device = next(model.parameters()).device
                if 'cuda' in device.type:
                    batch.to(device)

                outputs = model(batch, config)
                loss = model.loss(outputs, batch.labels, config)
                
                batch_size = batch.labels.size(0)
                train_total_loss += loss.item() * batch_size
                train_num_samples += batch_size
                
                loss.backward()
            
            # Calculate gradient norm
            grad_norm = 0.0
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm += param.grad.data.norm(2) ** 2
            grad_norm = torch.sqrt(grad_norm).item()

        else:
            with torch.no_grad():
                for batch in train_loader:
                    device = next(model.parameters()).device
                    if 'cuda' in device.type:
                        batch.to(device)

                    outputs = model(batch, config)
                    loss = model.loss(outputs, batch.labels, config)
                    
                    batch_size = batch.labels.size(0)
                    train_total_loss += loss.item() * batch_size
                    train_num_samples += batch_size

            grad_norm = None
        
        train_loss = train_total_loss / train_num_samples if train_num_samples > 0 else 0.0
        
        # Compute validation loss (always without gradients)
        val_total_loss = 0.0
        val_num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                device = next(model.parameters()).device
                if 'cuda' in device.type:
                    batch.to(device)

                outputs = model(batch, config)
                loss = model.loss(outputs, batch.labels, config)
                
                batch_size = batch.labels.size(0)
                val_total_loss += loss.item() * batch_size
                val_num_samples += batch_size
        
        val_loss = val_total_loss / val_num_samples if val_num_samples > 0 else 0.0
        
        return train_loss, val_loss, grad_norm
    # -------------------------------------------------------------------------------------------------- #

    print("Evaluating original losses and gradient norm L(θ*), ||∇L(θ*)||...")
    original_train_loss, original_val_loss, original_grad_norm = evaluate_loss_grad_norm(net, calc_grad_norm=True)
    print(f"Original train loss L_train(θ*): {original_train_loss:.6f}")
    print(f"Original val loss L_val(θ*): {original_val_loss:.6f}")
    print(f"Original gradient norm ||∇L(θ*)||: {original_grad_norm:.8f}")
    
    print("Computing original validation IoU...")
    original_mean_iou, original_per_class_ious = compute_validation_iou(net, val_loader, val_dataset, config)
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
 
        alpha = desired_percent / 100.0  # Direct scaling approach
        
        # Set perturbed weights: w* + alpha * direction
        trainable_params = [p for p in net.parameters() if p.requires_grad and len(p.size()) > 1]
        for (param, w, d) in zip(trainable_params, original_weights, directions):
            param.data.copy_(w + alpha * d)

        perturbed_train_loss, perturbed_val_loss, perturbed_grad_norm = evaluate_loss_grad_norm(net, calc_grad_norm=True)
        perturbed_train_losses.append(perturbed_train_loss)
        perturbed_val_losses.append(perturbed_val_loss)
        perturbed_grad_norms.append(perturbed_grad_norm)
        
        perturbed_mean_IoU, perturbed_per_class_IoU = compute_validation_iou(net, val_loader, val_dataset, config)
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


def setup_data_loaders(config):
    from torch.utils.data import DataLoader
    
    print("\nSetting up data loaders...")
    
    # Initialize dataset based on config.dataset
    if config.dataset == 'DALES':
        from datasets.DALES import DALESDataset, DALESSampler, DALESCollate
        
        training_dataset = DALESDataset(config, set='training', use_potentials=True)
        val_dataset = DALESDataset(config, set='validation', use_potentials=True)
        training_sampler = DALESSampler(training_dataset)
        val_sampler = DALESSampler(val_dataset)
        collate_fn = DALESCollate
                
    elif config.dataset == 'S3DIS':
        from datasets.S3DIS import S3DISDataset, S3DISSampler, S3DISCollate
        
        training_dataset = S3DISDataset(config, set='training', use_potentials=True)
        val_dataset = S3DISDataset(config, set='validation', use_potentials=True)
        training_sampler = S3DISSampler(training_dataset)
        val_sampler = S3DISSampler(val_dataset)
        collate_fn = S3DISCollate
                
    elif config.dataset == 'STPLS3D':
        from datasets.STPLS3D import STPLS3DDataset, STPLS3DSampler, STPLS3DCollate
        
        training_dataset = STPLS3DDataset(config, set='training', use_potentials=True)
        val_dataset = STPLS3DDataset(config, set='validation', use_potentials=True)
        training_sampler = STPLS3DSampler(training_dataset)
        val_sampler = STPLS3DSampler(val_dataset)
        collate_fn = STPLS3DCollate
        
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    training_loader = DataLoader(training_dataset,
                                batch_size=1,
                                sampler=training_sampler,
                                collate_fn=collate_fn,
                                num_workers=0,
                                pin_memory=True)
    
    val_loader = DataLoader(val_dataset,
                           batch_size=1,
                           sampler=val_sampler,
                           collate_fn=collate_fn,
                           num_workers=0,
                           pin_memory=True)
    
    print("Calibrating samplers...")
    training_sampler.calibration(training_loader, verbose=True)
    val_sampler.calibration(val_loader, verbose=True)

    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes: {training_dataset.num_classes}")
    print(f"Label values: {training_dataset.label_values}")
    print(f"Ignored labels: {training_dataset.ignored_labels}")
        
    return training_loader, val_loader, training_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Load and diagnose a trained KPConv model")
    parser.add_argument('--chosen_log', type=str, required=True, help="Path to the chosen log directory containing the trained model")
    parser.add_argument('--gpu', type=str, default='0', help="GPU ID to use (default: 0)")
    parser.add_argument('--chkp_idx', type=int, default=None, help="Index of checkpoint to load. If not specified, loads current checkpoint")
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

    training_loader = None
    val_loader = None

    try:

        # Find all checkpoints in the chosen training folder
        print(f"Log directory: {args.chosen_log}")
        chkp_path = os.path.join(args.chosen_log, 'checkpoints')
        
        if not os.path.exists(chkp_path):
            raise ValueError(f"Checkpoint directory not found: {chkp_path}")
        
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
        
        # Find which snapshot to restore
        if args.chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            if args.chkp_idx >= len(chkps):
                raise ValueError(f"Checkpoint index {args.chkp_idx} out of range. Available checkpoints: {len(chkps)}")
            chosen_chkp = np.sort(chkps)[args.chkp_idx]

        chosen_chkp = os.path.join(args.chosen_log, 'checkpoints', chosen_chkp)

        if not os.path.exists(chosen_chkp):
            raise ValueError(f"Checkpoint file not found: {chosen_chkp}")
        
        print(f"Loading checkpoint: {chosen_chkp}")
        
        # Load configuration and setup training dataset
        config = Config()
        config.load(args.chosen_log)
        
        print(f"Dataset: {config.dataset}")
        print(f"Number of classes: {config.num_classes}")

        training_loader, val_loader, training_dataset, val_dataset = setup_data_loaders(config)

        print("Initializing network...")
        t1 = time.time()
        net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)
        
        tester = ModelTester(net, chkp_path=chosen_chkp)
        print(f"Model loaded in {time.time() - t1:.1f}s")
        
        print_model_info(net, config)
        
        print(f"\nModel successfully loaded from epoch {tester.epoch}")
        print("Model is ready for diagnostics and analysis.")
        
        # Output directory (same as chosen_log directory)
        output_dir = args.chosen_log

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
                    val_loader=val_loader,
                    val_dataset=val_dataset,
                    config=config,
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
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Explicit cleanup to prevent memory leaks between back-to-back bash script calls
        if training_loader is not None:
            del training_loader
        if val_loader is not None:
            del val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
