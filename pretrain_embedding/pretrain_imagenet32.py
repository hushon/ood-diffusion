from typing import NamedTuple
import torch
from torch import nn, utils, optim, cuda
from torch.cuda import amp
import torchvision.transforms as T
import os
from tqdm import tqdm, trange
import argparse
import numpy as np
import random
from torch.nn.parallel import DataParallel
from torchvision import datasets
import atexit
from PIL import Image
from resnet_cifar100 import resnet18


torch.backends.cudnn.benchmark = True
SET_DETERMINISTIC = False

if SET_DETERMINISTIC:
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    np.random.seed(123)
    random.seed(123)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.485, 0.456, 0.406)
CIFAR100_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FLAGS(NamedTuple):
    DATA_ROOT = '/workspace/data'
    LOG_DIR = './weights'
    BATCH_SIZE = 256*2
    INIT_LR = 1E-3
    WEIGHT_DECAY = 1E-5
    MAX_EPOCH = 50
    N_WORKERS = 8
    SAVE = True


if FLAGS.SAVE:
    os.makedirs(FLAGS.LOG_DIR, exist_ok=True)


def tprint(obj):
    tqdm.write(obj.__str__())


class ImageNet32(datasets.VisionDataset):
    train_list = (
        'Imagenet32_train_npz/train_data_batch_1.npz',
        'Imagenet32_train_npz/train_data_batch_2.npz',
        'Imagenet32_train_npz/train_data_batch_3.npz',
        'Imagenet32_train_npz/train_data_batch_4.npz',
        'Imagenet32_train_npz/train_data_batch_5.npz',
        'Imagenet32_train_npz/train_data_batch_6.npz',
        'Imagenet32_train_npz/train_data_batch_7.npz',
        'Imagenet32_train_npz/train_data_batch_8.npz',
        'Imagenet32_train_npz/train_data_batch_9.npz',
        'Imagenet32_train_npz/train_data_batch_10.npz',
    )
    val_list = (
        'Imagenet32_val_npz/val_data.npz',
    )
    train_cache_file = 'train_data.npz'
    val_cache_file = 'val_data.npz'
    MEAN = (0.4811, 0.4575, 0.4079)
    STD = (0.2604, 0.2532, 0.2682)

    def __init__(self, root, transform=None, target_transform=None, train=True):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        if not os.path.exists(os.path.join(self.root, self.train_cache_file)):
            self._create_cache(self.train_list, os.path.join(self.root, self.train_cache_file))
        if not os.path.exists(os.path.join(self.root, self.val_cache_file)):
            self._create_cache(self.val_list, os.path.join(self.root, self.val_cache_file))

        entry = np.load(os.path.join(self.root, self.train_cache_file if self.train else self.val_cache_file))
        self.data = entry['data']
        self.targets = entry['labels']

    def _create_cache(self, file_list, cache_filepath):
        data = []
        targets = []
        for file_name in file_list:
            file_path = os.path.join(self.root, file_name)
            entry = np.load(file_path)
            data.append(entry['data'])
            targets.append(entry['labels'])
        data = np.concatenate(data, axis=0).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        targets = np.concatenate(targets, axis=0) - 1
        np.savez(cache_filepath, data=data, labels=targets)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.targets)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)



def train():

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.ColorJitter(0.1, 0.1, 0.1),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(ImageNet32.MEAN, ImageNet32.STD)
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(ImageNet32.MEAN, ImageNet32.STD)
    ])

    train_dataset = ImageNet32(os.path.join(FLAGS.DATA_ROOT, 'Imagenet32'), transform=transform_train, train=True)
    test_dataset = ImageNet32(os.path.join(FLAGS.DATA_ROOT, 'Imagenet32'), transform=transform_test, train=False)

    train_loader = MultiEpochsDataLoader(train_dataset,
            batch_size=FLAGS.BATCH_SIZE,
            shuffle=True,
            num_workers=FLAGS.N_WORKERS,
            )
    test_loader = MultiEpochsDataLoader(test_dataset,
            batch_size=FLAGS.BATCH_SIZE*2,
            shuffle=False,
            num_workers=FLAGS.N_WORKERS
            )

    model = resnet18(num_classes=1000)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load("./resnet18_cifar100_imagenet32.pt", map_location="cpu"), strict=False)
    print(f"{missing_keys=}\n{unexpected_keys=}")
    model = DataParallel(model).cuda()


    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.INIT_LR, momentum=0.9, weight_decay=FLAGS.WEIGHT_DECAY)
    # lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS.INIT_LR, epochs=FLAGS.MAX_EPOCH, steps_per_epoch=len(train_loader), pct_start=0.1)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, FLAGS.MAX_EPOCH*len(train_loader))

    def train_epoch():
        model.train()
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda()
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target).mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if i%1000 == 0:
                tprint(f'[TRAIN][{i}/{len(train_loader)}] LR {lr_scheduler.get_last_lr()[-1]:.2e} | loss {loss.cpu().item():.3f}')

    @torch.no_grad()
    def evaluate():
        losses = []
        corrects = []
        model.eval()
        for input, target in test_loader:
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            losses.append(loss.view(-1))
            corrects.append((target == output.max(-1).indices).view(-1))
        avg_loss = torch.cat(losses).mean().item()
        avg_acc = torch.cat(corrects).float().mean().item()*100
        tprint(f'[TEST] loss {avg_loss:.3f} | T1acc {avg_acc:.2f}')

    def save_pickle():
        pickle = model.module.state_dict()
        # pickle.pop('fc.weight')
        # pickle.pop('fc.bias')
        pickle_path = os.path.join(FLAGS.LOG_DIR, f'state_dict.pt')
        torch.save(pickle, pickle_path)
        tprint(f'[SAVE] Saved to {pickle_path}')

    if FLAGS.SAVE: atexit.register(save_pickle)

    pbar = trange(FLAGS.MAX_EPOCH, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', smoothing=1., dynamic_ncols=True)

    for epoch in pbar:
        train_epoch()
        evaluate()
        save_pickle()

    if FLAGS.SAVE:
        save_pickle()

if __name__ == "__main__":
    train()