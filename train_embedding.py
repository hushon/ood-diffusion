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
#             transforms.RandomHorizontalFlip(),
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


# transform_svhn = transforms.Compose([
#             transforms.Resize(32),
#             transforms.CenterCrop((32,32)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4811, 0.4575, 0.4079), (0.2604, 0.2532, 0.2682))
#             # transforms.Lambda(lambda t: (t * 2) - 1)
#         ])

transform_svhn = transforms.Compose([
            # transforms.Resize(32),
            # transforms.CenterCrop((32,32)),
            transforms.Resize(224),
            transforms.CenterCrop((224,224)),
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

# dataset = CIFAR10(DATA_ROOT, transform=transform, train=True, download=True)
dataset = CIFAR100(DATA_ROOT, transform=transform, train=True, download=True)
# dataset = CelebA(DATA_ROOT, transform=transform, download=True)
# dataset = SVHN(DATA_ROOT, split="train", transform=transform_svhn, download=True)

net = Unet(
    dim = 512,
    dim_mults = (1, 2, 4, 8)
)
# net = torch.jit.script(net)
# net = torch.jit.trace(net, (torch.rand((1, 3, 32, 32)).cuda(), torch.randint(0, 10, size=(1,)).cuda()))
# net = net.cuda()

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
    # train_batch_size = 512,
    train_batch_size = 1024,
    train_lr = 1e-4,
    train_num_steps = 50000,         # total training steps
    # gradient_accumulate_every = 1,    # gradient accumulation steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    save_and_sample_every = 500,
    fp16 = True,                       # turn on mixed precision training with apex
    results_folder = RESULTS_FOLDER
)

trainer.train()
