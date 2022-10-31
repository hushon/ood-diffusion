import json
import argparse

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision import datasets, transforms
from datetime import datetime
import pathlib

class PathListDataset(torch.utils.data.Dataset):
    def __init__(self, path_list, transform=None):
        super().__init__()
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path = self.path_list[index]
        img = datasets.folder.default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, choices=['train', 'test'], default='train', required=True)
    parser.add_argument('--log_dir', type=str, default='./results/')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--input_json_path', type=str, default='sample_input.json', required=True)
    parser.add_argument('--output_json_path', type=str, default='sample_output.json')
    args = parser.parse_args()


    TIMESTAMP = datetime.now()
    RESULTS_FOLDER = pathlib.Path(f"{args.log_dir}/{TIMESTAMP:%Y%m%d-%H%M%S}")
    RESULTS_FOLDER.mkdir()


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


    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4811, 0.4575, 0.4079), (0.2604, 0.2532, 0.2682))
            # transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    if args.phase == 'train':
        with open(args.input_json_path, "r") as fp:
            train_images = json.load(fp)["train_images"]
        dataset = PathListDataset(train_images, transform)
    elif args.phase == 'test':
        with open(args.input_json_path, "r") as fp:
            test_images = json.load(fp)["test_images"]
        dataset = PathListDataset(test_images, transform)


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


    if args.phase == 'train':
        trainer.train()

    elif args.phase == 'test':
        trainer.load(args.ckpt_path)            # load model
        collect = trainer.validate().cpu().tolist()
        paths = dataset.path_list
        output = {k: v for k, v in zip(paths, collect)}
        with open(args.output_json_path, "w") as fp:
            json.dump(output, fp, indent=4)




