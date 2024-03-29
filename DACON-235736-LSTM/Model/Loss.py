import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss
