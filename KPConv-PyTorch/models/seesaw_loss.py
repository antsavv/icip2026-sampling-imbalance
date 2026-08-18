import torch
import torch.nn as nn
import numpy as np

class SeesawLoss(nn.Module):
    """
    Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    
    Args:
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor. Defaults to 0.8.
        q (float): The ``q`` in the compensation factor. Defaults to 2.0.
        eps (float): The minimal value of divisor to smooth the computation 
                    of compensation factor. Defaults to 1e-2.
        weight (torch.Tensor or None): Optional class weights for cross-entropy.
    """
    def __init__(self, num_classes, p=0.8, q=2.0, eps=1e-2, weight=None):
        super(SeesawLoss, self).__init__()
        
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        self.weight = weight
        
        # Register buffer for cumulative samples tracking
        self.register_buffer('cum_samples', torch.zeros(num_classes, dtype=torch.float))
        
        print("-------------------------------------------------------------------------------")
        print("SeesawLoss initialized with num_classes:", num_classes, "p:", p, "q:", q, "eps:", eps)
        print("-------------------------------------------------------------------------------")

    def forward(self, logits, target):
        N, C = logits.shape[:2]

        if logits.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logits = logits.view(N, C, -1)
            logits = logits.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logits = logits.view(-1, logits.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        target = target.view(-1)  # [N,d1,d2,...]->[N*d1*d2*...]

        mask = (target != -1)
        logits = logits[mask]  # logits: [N_valid, C] keep only valid samples (rows)
        target = target[mask]  # target: [N_valid]

        if len(target) == 0:  # Handle edge case where no valid targets
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        # Update cumulative samples for each class in current batch
        unique_labels = target.unique()
        for u_l in unique_labels:
            if 0 <= u_l < self.num_classes:
                inds_ = target == u_l
                self.cum_samples[u_l] += inds_.sum()

        # Create one-hot labels and initialize seesaw weights
        onehot_labels = torch.nn.functional.one_hot(target.long(), num_classes=self.num_classes).float()
        seesaw_weights = torch.ones_like(onehot_labels)

        # Mitigation factor computation
        if self.p > 0:
            cum_samples_clamped = self.cum_samples.clamp(min=1)
            sample_ratio_matrix = cum_samples_clamped[None, :] / cum_samples_clamped[:, None]
            index = (sample_ratio_matrix < 1.0).float()
            sample_weights = sample_ratio_matrix.pow(self.p) * index + (1 - index)
            mitigation_factor = sample_weights[target.long(), :]
            seesaw_weights = seesaw_weights * mitigation_factor

        # Compensation factor computation  
        if self.q > 0:
            scores = torch.nn.functional.softmax(logits.detach(), dim=1)
            self_scores = scores[
                torch.arange(0, len(scores)).to(scores.device).long(), 
                target.long()]
            score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor

        # Apply seesaw weights to logits
        adjusted_logits = logits + (seesaw_weights.log() * (1 - onehot_labels))

        # Compute cross-entropy loss
        return torch.nn.functional.cross_entropy(adjusted_logits, target, weight=self.weight)