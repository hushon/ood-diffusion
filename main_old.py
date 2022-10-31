import json
import argparse
import torch
from torch import nn
import torch.utils.data
from torchvision import datasets, transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

INPUT_JSON = "./sample_input.json"
OUTPUT_JSON = "./sample_output.json"


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


class OODDetector:
    def __init__(self) -> None:
        super().__init__()
        self.model: nn.Module
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4811, 0.4575, 0.4079), (0.2604, 0.2532, 0.2682))
            # transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def load(self):
        trainer.load("./results/20220926-033403/model-49500.pt") # TRAIN=CIFAR100

    def fit(self, input_json: str) -> None:
        with open(input_json, "r") as fp:
            train_images = json.load(fp)["train_images"]
        dataset = PathListDataset(train_images, self.transform)

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

        trainer.train()

    def predict(self, input_json: str) -> dict:

        with open(input_json, "r") as fp:
            test_images = json.load(fp)["test_images"]

        dataset = PathListDataset(test_images, self.transform)

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

        collect = trainer.validate().cpu().tolist()
        paths = dataset.path_list

        output = {k: v for k, v in zip(paths, collect)}
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', type=str, choices=['train', 'test'], default='train', required=True)
    parser.add_argument('--json_path', type=str, default='sample_input.json', required=True)
    args = parser.parse_args()

    ood_detector = OODDetector()
    output = ood_detector.predict(INPUT_JSON)

    with open(OUTPUT_JSON, "w") as fp:
        json.dump(output, fp, indent=4)