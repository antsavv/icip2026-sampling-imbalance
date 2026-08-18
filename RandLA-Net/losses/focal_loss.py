# License from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/tree/master?tab=Apache-2.0-1-ov-file

import torch
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    """
    This is an implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Modified version from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/focal_loss.py (FocalLoss_Ori)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        mode: 'normal', 'quantile', 'non-deterministic'
        reduction:
    """

    def __init__(self, num_class, alpha=1.0, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')


    def forward(self, logit, target):
        """
        Args:
            logit: [N, C] tensor of logits
            target: [N] tensor of targets
        """
        alpha = self.alpha.to(logit.device)

        # prob: [N, C] - predicted probabilities for each class
        prob = torch.softmax(logit, dim=1)

        # Get probability of the true class for each sample
        target_idx = target.view(-1, 1)  # [N, 1]
        prob_true = prob.gather(1, target_idx).view(-1)  # [N]
        prob_true = torch.clamp(prob_true, min=1e-8, max=1.0)  # Numerical stability

        logpt = torch.log(prob_true)
        alpha_class = alpha[target.long()]
        class_weight = -alpha_class * torch.pow(1.0 - prob_true, self.gamma)
        loss = class_weight * logpt

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss