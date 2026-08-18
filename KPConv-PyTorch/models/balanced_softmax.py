import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class BalancedSoftmax(nn.Module):
    """
    Balanced Softmax Loss for addressing class imbalance.
        
    Args:
        cls_num_list (list or tensor): List or tensor containing the number of samples
            for each class. The length should match the number of classes in the dataset.
    
    Attributes:
        sample_per_class (torch.Tensor): Buffer storing the sample count per class as a tensor.
    """

    def __init__(self, cls_num_list):
        super(BalancedSoftmax, self).__init__()

        self.register_buffer('sample_per_class', torch.tensor(cls_num_list, dtype=torch.float32))

        print("-------------------------------------------------------------------------------")
        print("BalancedSoftmaxLoss initialized with self.sample_per_class:", self.sample_per_class)
        print("-------------------------------------------------------------------------------")

    def forward(self, logits, target):
        N, C = logits.shape[:2]
        
        # Handle multi-dimensional inputs (for segmentation tasks)
        if logits.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logits = logits.view(N, C, -1)
            logits = logits.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logits = logits.view(-1, logits.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
            
        target = target.view(-1)  # [N,d1,d2,...]->[N*d1*d2*...,]

        if len(target) == 0:  # Handle edge case where no valid targets
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        # Filter out ignored classes (-1 values)
        mask = (target != -1)
        logits = logits[mask]
        target = target[mask]

        # Apply balanced softmax loss
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        return F.cross_entropy(input=logits, target=target, reduction='mean')