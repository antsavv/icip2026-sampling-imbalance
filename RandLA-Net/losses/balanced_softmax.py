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
        """
        Args:
            logits: [N, C] tensor of logits
            target: [N] tensor of targets
        """
        if len(target) == 0:  # Handle edge case where no valid targets
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        # Apply balanced softmax loss
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        return F.cross_entropy(input=logits, target=target, reduction='mean')