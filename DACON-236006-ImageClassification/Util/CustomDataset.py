from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        super().__init__()

        self.image_paths = image_paths
        self.labels = labels

        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_path = './Data/' + image_path

        image = read_image(image_path, mode=ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.image_paths)
