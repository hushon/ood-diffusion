import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

import torch.utils.data
from pathlib import Path
from torchvision import utils as vutils

import numpy as np
from tqdm import tqdm, trange
from einops import rearrange

from torch.cuda import amp
from torch.utils.tensorboard.writer import SummaryWriter
# from pretrain_embedding.resnet_cifar100 import resnet18
from torchvision.models import resnet18

torch.backends.cudnn.benchmarks = True

# helpers functions

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def cycle(data_loader):
    while True:
        for data in data_loader:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


@torch.jit.script
def update_ema(running: torch.Tensor, new: torch.Tensor, beta: torch.Tensor):
    running.sub_(new).mul_(beta).add_(new)


class EMA(nn.Module):
    def __init__(self, model: nn.Module, beta=0.995):
        super().__init__()
        self.running_model = model
        self.ema_model = copy.deepcopy(self.running_model)
        self.ema_model.eval()
        self.register_buffer("beta", torch.tensor(beta))

    def update(self):
        # for running_params, ema_params in zip(self.running_model.parameters(), self.ema_model.parameters()):
        #     ema_params.data.sub_(running_params.data).mul_(self.beta).add_(running_params.data)
        #     # update_ema(running_params.data, ema_params.data, self.beta)
        # for running_buf, ema_buf in zip(self.running_model.buffers(), self.ema_model.buffers()):
        #     ema_buf.data.sub_(running_buf.data).mul_(self.beta).add_(running_buf.data)
        #     # update_ema(running_buf.data, ema_buf.data, self.beta)
        self.ema_model = self.running_model


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(input, repeat=False):
    repeat_noise = lambda: torch.randn_like((1, *input[1:])).repeat(input[0], *((1,) * (len(input) - 1)))
    noise = lambda: torch.randn_like(input)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        dims = 512,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None
    ):
        super().__init__()
        self.dims = dims
        self.denoise_fn = denoise_fn

        # self.feat_net = resnet18(num_classes=0).eval()
        # missing_keys, unexpected_keys = self.feat_net.load_state_dict(torch.load("./pretrain_embedding/weights/state_dict.pt", map_location='cpu'), strict=False)
        # print(f"{missing_keys=}\n{unexpected_keys=}")
        self.feat_net = resnet18(pretrained=True).eval().cuda()
        self.feat_net.fc = torch.nn.Identity()
        for param in self.feat_net.parameters():
            param.requires_grad_(False)
        self.feat_net = torch.jit.script(self.feat_net)


        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b = x.size(0)
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps, leave=False):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        dims = self.dims
        return self.p_sample_loop((batch_size, dims))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        # b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start)) # neat trick!

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean(list(range(x_recon.ndim))[1:])
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon, reduction="none").mean(list(range(x_recon.ndim))[1:])
        else:
            raise NotImplementedError()

        return loss

    # def forward(self, x, *args, **kwargs):
    #     b = x.size(0)
    #     device = x.device
    #     t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
    #     with torch.no_grad():
    #         x = self.feat_net(x)
    #     return self.p_losses(x, t, *args, **kwargs)

    # def compute_loglikelihood(self, x):
    def forward(self, x):
        x = self.feat_net(x)

        b = x.size(0)
        l = []
        for t in range(self.num_timesteps):
            # t = torch.full(size=(b,), fill_value=t, device=x.device, dtype=torch.long)
            t = torch.full(size=(b,), fill_value=100, device=x.device, dtype=torch.long)
            losses = self.p_losses(x, t)
            l.append(losses)
        return torch.mean(torch.stack(l, dim=0), dim=0)



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


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        dataset: torch.utils.data.Dataset,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        save_and_sample_every = 1000,
        results_folder = './results'
    ):
        self.model = diffusion_model
        self.ema = EMA(self.model, ema_decay)

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.dataset = dataset
        self.data_loader = MultiEpochsDataLoader(self.dataset, batch_size=train_batch_size, num_workers=8*torch.cuda.device_count(), shuffle=True)
        self.data_loader_cycle = cycle(self.data_loader)
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr, weight_decay=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.train_num_steps)

        self.step = 0
        self.fp16 = fp16

        self.results_folder = Path(results_folder)
        print(f"{self.results_folder=}")
        self.results_folder.mkdir(parents=True, exist_ok = True)
        self.summary_writer = SummaryWriter(self.results_folder)
        # self.summary_writer.add_image("sample_batch", vutils.make_grid(next(self.data_loader_cycle))[:36])

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.ema.running_model.state_dict(),
            'ema': self.ema.ema_model.state_dict()
        }
        torch.save(data, self.results_folder/f'model-{milestone}.pt')

    def load(self, path):
        checkpoint = torch.load(path)
        self.step = checkpoint["step"]
        self.ema.running_model.load_state_dict(checkpoint["model"])
        self.ema.ema_model.load_state_dict(checkpoint["ema"])
        print(f"Restored trainer state from {path}")

    def train(self):
        self.model.train()
        grad_scaler = amp.GradScaler(enabled=self.fp16)
        dp_model = nn.parallel.DataParallel(self.model)
    
        for _ in (pbar := trange(self.step, self.train_num_steps)):
            for _ in range(self.gradient_accumulate_every):
                data = next(self.data_loader_cycle)
                # data = data.cuda(non_blocking=True)
                with amp.autocast(enabled=self.fp16):
                    # loss = self.model(data).mean()
                    loss = dp_model(data).mean()
                grad_scaler.scale(loss / self.gradient_accumulate_every).backward()

            grad_scaler.step(self.optimizer)
            grad_scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            # self.lr_scheduler.step()

            self.ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                # batches = num_to_groups(36, self.batch_size)
                # all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                # all_images = torch.cat(all_images_list, dim=0)
                # all_images = (all_images + 1) * 0.5
                # vutils.save_image(all_images, self.results_folder / f'sample-{self.step}.png', nrow = 6)
                # tqdm.write(f"saved to {str(self.results_folder / f'sample-{self.step}.png')}")
                # self.summary_writer.add_images("samples", all_images, self.step)
                self.save(self.step)

            if self.step>0 and self.step%10==0:
                loss = loss.item()
                pbar.set_postfix({
                    "loss": loss,
                    "img/s": f"{pbar.format_dict['rate']*self.batch_size*self.gradient_accumulate_every:.0f}"
                })
                self.summary_writer.add_scalar("loss", loss, self.step)

            self.step += 1

        print('training completed')

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        # compute nll on validation set
        dp_model = nn.parallel.DataParallel(self.ema.ema_model)
        collect = []
        for data in tqdm(self.data_loader, desc="validation batches"):
            data = data.cuda()
            with amp.autocast(enabled=self.fp16):
                # losses = self.model.compute_loglikelihood(data)
                losses = dp_model(data)
            collect.append(losses)
        collect = torch.cat(collect, 0)
        print(collect)
        torch.save(collect, self.results_folder/"validation_nll.pth")