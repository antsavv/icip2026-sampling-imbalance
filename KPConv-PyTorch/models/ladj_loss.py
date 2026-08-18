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

        # Logit adjustment: add bias to all logits
        logits_adj = logits + self.bias.to(logits.device)

        # Cross-entropy on adjusted logits
        return nn.functional.cross_entropy(logits_adj, target, weight=self.weight)