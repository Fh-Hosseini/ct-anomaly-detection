import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, labels):
        cross_entropy_per_sample = F.cross_entropy(outputs, labels, weight=self.alpha, reduction="none")
        probability_of_true_class = torch.exp(-cross_entropy_per_sample)
        focal_loss_per_sample = ((1 - probability_of_true_class) ** self.gamma) * cross_entropy_per_sample
        return focal_loss_per_sample.mean()