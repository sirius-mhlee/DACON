import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features, targets, window_size):
        super().__init__()
        
        self.features = features
        self.targets = targets

        self.window_size = window_size

    def __getitem__(self, index):
        feature = torch.tensor(self.features[index:index + self.window_size, :], dtype=torch.float32)

        if self.targets is not None:
            target = torch.tensor(self.targets[index + self.window_size], dtype=torch.float32)
            return feature, target
        else:
            return feature

    def __len__(self):
        return len(self.features) - self.window_size
