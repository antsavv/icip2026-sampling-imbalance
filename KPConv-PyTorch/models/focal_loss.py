# License from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/tree/master?tab=Apache-2.0-1-ov-file

import torch
import torch.nn as nn
import numpy as np


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, probs, targets):
#         # probs: predicted probabilities (batch_size, num_classes)
#         # targets: class labels (batch_size,)
#
#         # Compute cross entropy
#         ce_loss = nn.functional.cross_entropy(probs, targets, reduction='none')
#
#         # Compute focal loss weights
#         if self.alpha is not None:
#             alpha = self.alpha.unsqueeze(0).to(targets.device)
#             if alpha.dim() == 1:    # alpha is a single value: shape: (1, )
#                 alpha_factor = alpha
#             elif alpha.dim() == 2:   # alpha is a list of weights: shape: (1, num_classes)
#                 alpha_factor = alpha[:, targets]  # apply alpha to targets of corresponding class
#             else:
#                 raise ValueError("Dimension of 'alpha' is larger than 2.")
#         else:
#             alpha_factor = 1
#
#         # pt = torch.exp(-ce_loss)
#         # focal_weights = focal_weights * (1 - pt) ** self.gamma
#
#         temp = probs.softmax(dim=1)
#         if temp.min() < 0 or temp.max() > 1:
#             sys.exit("Probabilities are outside the range [0, 1]!")
#         if abs(temp.sum() - temp.shape[2]) > 0.01:
#             print(f'Difference = {abs(temp.sum() - temp.shape[2])}')
#             sys.exit("Probabilities do not sum to 1!")
#
#         modulating_factor = (1 - temp) ** self.gamma
#
#         loss = alpha_factor * modulating_factor * ce_loss
#
#         # Apply reduction
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#
#         return loss

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

    def __init__(self, num_class, alpha=0.25, gamma=2, mode='normal', reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.mode = mode
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
        N, C = logit.shape[:2]

        alpha = self.alpha.to(logit.device)

        # prob is a 2D tensor with shape [batch_size, num_classes]. Each row contains the predicted probabilities for each class for a specific sample.
        prob = torch.softmax(logit, dim=1)  # Apply softmax to each row
        # prob = torch.sigmoid(logit)

        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # Ignore the -1 values in the target tensor (ignored classes).
        mask = (target != -1).view(-1)
        prob = prob[mask]

        # target_masked is a 2D tensor with shape [batch_size, 1]. 
        # Each element in target_masked is the true class label of a specific sample.
        target_masked = target[mask].view(-1, 1)

        # prob.gather(1, target_masked) is using target_masked as indices to select one element from each row of prob. 
        # The output is a 2D tensor with shape [batch_size, 1], where each element is the predicted probability of the "true class" of each sample. 
        # .view(-1) reshapes this tensor into a 1D tensor with shape [batch_size].

        prob_true = prob.gather(1, target_masked).view(-1)
        prob_true = torch.clamp(prob_true, min=1e-8, max=1.0)  # Numerical stability

        logpt = torch.log(prob_true)
        alpha_class = alpha[target_masked.squeeze().long()]
        class_weight = -alpha_class * torch.pow(1.0 - prob_true, self.gamma)
        loss = class_weight * logpt
        
        if self.mode == 'quantile':
            # Quantile method: Consider mis-classified samples by focusing loss calculation on the hardest examples
            # (considering that lower probabilities correspond to harder examples).
            loss = loss[prob < torch.quantile(prob, 0.2)]
        elif self.mode == 'non-deterministic':
            # Non-deterministic method: randomly choose b% of points in majority classes to remove from loss
            mask = torch.ones(target.numel(), dtype=torch.bool)
            ratio = 0.95  # Amount of points to remove
            for i in [1, 2, 8]:  # Loop over majority classes
                class_mask = (target.squeeze() == i).float()
                class_sum = class_mask.sum()
                if class_sum > (1/ratio):  # If there are points in the batch select "ratio" of them to remove
                    class_indices = class_mask.multinomial(torch.round(class_sum * ratio).int(), replacement=False)
                    mask[class_indices] = False
                    # print(f"Loss {i}: {loss.shape} and class_mask: {torch.round(class_sum * ratio).int()}")
            loss = loss[mask]
        elif self.mode == 'normal':
            pass
        else:
            raise ValueError(f"Mode {self.mode} not recognized.")

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss