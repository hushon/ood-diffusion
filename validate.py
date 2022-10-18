import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision import datasets, transforms
from datetime import datetime
import pathlib


TIMESTAMP = datetime.now()

DATA_ROOT = pathlib.Path('/ramdisk/')
RESULTS_FOLDER = pathlib.Path(f"./results/{TIMESTAMP:%Y%m%d-%H%M%S}")
RESULTS_FOLDER.mkdir()

# dataset classes

# transform = transforms.Compose([
#             transforms.Resize(32),
#             transforms.CenterCrop((32,32)),
#             # transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4811, 0.4575, 0.4079), (0.2604, 0.2532, 0.2682))
#             # transforms.Lambda(lambda t: (t * 2) - 1)
#         ])
transform = transforms.Compose([
            # transforms.Resize(32),
            # transforms.CenterCrop((32,32)),
            transforms.Resize(224),
            transforms.CenterCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4811, 0.4575, 0.4079), (0.2604, 0.2532, 0.2682))
            # transforms.Lambda(lambda t: (t * 2) - 1)
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

# dataset = CIFAR10(DATA_ROOT, transform=transform, train=False, download=True)
dataset = CIFAR100(DATA_ROOT, transform=transform, train=False, download=True)
# dataset = CelebA(DATA_ROOT, transform=transform, download=True, split='valid')
# dataset = SVHN(DATA_ROOT, split="test", transform=transform, download=True)


net = Unet(
    dim = 512,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion_model = GaussianDiffusion(
    net,
    dims=512,
    timesteps = 1000,   # number of steps
    loss_type = 'l2'    # L1 or L2
).cuda()


trainer = Trainer(
    diffusion_model,
    dataset,
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

# trainer.train()

# trainer.load("./results/20220701-092117/model-99500.pt") # TRAIN = CelebA
# trainer.load("./results/20220725-052651/model-99500.pt") # TRAIN = CIFAR100
# trainer.load("./results/20220725-053137/model-99500.pt") # TRAIN = CIFAR10
# trainer.load("./results/20220725-080741/model-99500.pt") # TRAIN = SVHN
# trainer.load("./results/20220922-115306/model-9500.pt") # TRAIN=CIFAR100
# trainer.load("./results/20220923-055941/model-9500.pt") # TRAIN=CIFAR100
# trainer.load("./results/20220924-045219/model-45000.pt") # TRAIN=SVHN
# trainer.load("./results/20220926-032602/model-49500.pt") # TRAIN=CIFAR10
trainer.load("./results/20220926-033403/model-49500.pt") # TRAIN=CIFAR100
trainer.validate()
