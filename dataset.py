import torch
import torch.utils.data
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image


class ImageGlobDataset(torch.utils.data.Dataset):
    default_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __init__(self, folder, transform = None, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img


class CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image

class CIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image

class CelebA(datasets.CelebA):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image

class SVHN(datasets.SVHN):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img
