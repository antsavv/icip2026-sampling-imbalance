import torch
import torch.nn as nn
import numpy as np

class LADJLoss(nn.Module):
    """
    Logit Adjustment Loss
    Args:
        cls_num_list (list or np.array): Number of samples for each class.
        tau (float): Temperature parameter for logit adjustment.
        weight (torch.Tensor or None): Optional class weights for cross-entropy.
    """

    def __init__(self, cls_num_list, tau=1.0, weight=None):
        super(LADJLoss, self).__init__()

        cls_num_list = np.array(cls_num_list)

        prior = cls_num_list / cls_num_list.sum()  # Compute class priors
        
        bias = tau * torch.log(torch.tensor(prior, dtype=torch.float32))  # Logit adjustment bias for each class
        self.register_buffer('bias', bias)
        self.weight = weight

        print("-------------------------------------------------------------------------------")
        print("LogitAdjustmentLoss initialized with class priors:", prior)
        print("LogitAdjustment bias per class:", self.bias.cpu().numpy())
        print("-------------------------------------------------------------------------------")


    def forward(self, logits, target):
        """
        Args:
            logits: [N, C] tensor of logits
            target: [N] tensor of targets
        """
        # Logit adjustment: add bias to all logits
        logits_adj = logits + self.bias.to(logits.device)

        # Cross-entropy on adjusted logits
        return nn.functional.cross_entropy(logits_adj, target, weight=self.weight)