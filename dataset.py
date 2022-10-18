import torch
import torch.utils.data
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image


class CelebA(datasets.CelebA):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image
