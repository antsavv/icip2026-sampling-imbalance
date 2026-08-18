import torch
import torch.nn as nn
import numpy as np

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (LDAM)
    Args:
        cls_num_list (list or np.array): Number of samples for each class.
        max_m (float): Maximum margin.
        weight (torch.Tensor or None): Optional class weights for cross-entropy.
        s (float): Scaling factor for logits.
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):

        print("-------------------------------------------------------------------------------")
        print("LDAMLoss initialized with cls_num_list:", cls_num_list, "max_m:", max_m, "s:", s)
        print("-------------------------------------------------------------------------------")

        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float32)
        self.register_buffer('m_list', m_list)
        assert s > 0
        self.s = s
        self.weight = weight

        print("LDAM margins per class:", self.m_list.cpu().numpy())

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

        # Prepare margin for each sample in the batch
        margin = self.m_list[target].to(logits.device)  # [N_valid]

        # Create a copy of logits to subtract margin from the true class logit
        logits_m = logits.clone()
        logits_m[torch.arange(logits.size(0)), target] -= margin  # logits_m[i, target[i]] -= margin[i]

        # Scale logits
        logits_m = self.s * logits_m

        # Cross-entropy on margin-adjusted logits
        return nn.functional.cross_entropy(logits_m, target, weight=self.weight)