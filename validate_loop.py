import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision import datasets, transforms
from datetime import datetime
import pathlib



DATA_ROOT = pathlib.Path('/ramdisk/')

# dataset classes

transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop((32,32)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

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


net = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion_model = GaussianDiffusion(
    net,
    image_size = 32,
    timesteps = 1000,   # number of steps
    loss_type = 'l2'    # L1 or L2
).cuda()



checkpoint_paths = [
    "./results/20220725-052651/model-99500.pt", # TRAIN = CIFAR100
    "./results/20220725-053137/model-99500.pt", # TRAIN = CIFAR10
    "./results/20220725-080741/model-99500.pt" # TRAIN = SVHN
]

test_datasets = [
    CIFAR10(DATA_ROOT, transform=transform, train=False),
    CIFAR100(DATA_ROOT, transform=transform, train=False),
    SVHN(DATA_ROOT, split="test", transform=transform, download=True),
]

for ckpt in checkpoint_paths:
    for dset in test_datasets:

        TIMESTAMP = datetime.now()
        RESULTS_FOLDER = pathlib.Path(f"./results/{TIMESTAMP:%Y%m%d-%H%M%S}")
        RESULTS_FOLDER.mkdir()

        trainer = Trainer(
            diffusion_model,
            dset,
            # train_batch_size = 32,
            train_batch_size = 1024,
            train_lr = 1e-4,
            train_num_steps = 100000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            save_and_sample_every = 500,
            fp16 = True,                       # turn on mixed precision training with apex
            results_folder = RESULTS_FOLDER
        )

        trainer.load(ckpt)
        trainer.validate()
