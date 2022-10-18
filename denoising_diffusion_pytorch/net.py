import math
import torch
from torch import nn
from einops import rearrange
from inspect import isfunction
from einops.layers.torch import Rearrange

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

# class LayerNorm(nn.Module):
#     def __init__(self, dim, eps = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
#         self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

#     def forward(self, x):
#         std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
#         mean = torch.mean(x, dim = 1, keepdim = True)
#         return (x - mean) / (std + self.eps) * self.g + self.b


class LayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(input.permute(0,2,3,1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0,3,1,2)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim_out),
            # nn.GroupNorm(groups, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)

        h += self.mlp(time_emb)

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        # q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        q, k, v = qkv.view(b, 3, self.heads, -1, h*w).permute(1, 0, 2, 3, 4).unbind(0)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        # out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        out = out.view(b, -1, h, w)
        return self.to_out(out)

# model
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 512,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        # dims = [channels, *map(lambda m: dim * m, dim_mults)]
        # in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            raise ValueError

        # self.downs = nn.ModuleList([])
        # self.ups = nn.ModuleList([])
        # num_resolutions = len(in_out)

        # for ind, (dim_in, dim_out) in enumerate(in_out):
        #     is_last = ind >= (num_resolutions - 1)

        #     self.downs.append(nn.ModuleList([
        #         ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim),
        #         ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim),
        #         Residual(PreNorm(dim_out, LinearAttention(dim_out))),
        #         Downsample(dim_out) if not is_last else nn.Identity()
        #     ]))

        # mid_dim = dims[-1]
        # self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        # self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        # self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        #     is_last = ind >= (num_resolutions - 1)

        #     self.ups.append(nn.ModuleList([
        #         ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
        #         ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
        #         Residual(PreNorm(dim_in, LinearAttention(dim_in))),
        #         Upsample(dim_in) if not is_last else nn.Identity()
        #     ]))

        # out_dim = default(out_dim, channels)
        # self.final_conv = nn.Sequential(
        #     Block(dim, dim),
        #     nn.Conv2d(dim, out_dim, 1)
        # )

        self.linear_layers = nn.ModuleList([ResnetBlock(dim, dim, time_emb_dim=time_dim) for _ in range(32)])
        self.final_linear = nn.Sequential(Block(dim,dim), nn.Linear(dim, dim))

    # def forward(self, x, time):
    #     t = self.time_mlp(time)

    #     h = []

    #     for modulelist in self.downs:
    #         x = modulelist[0](x, t)
    #         x = modulelist[1](x, t)
    #         x = modulelist[2](x)
    #         h.append(x)
    #         x = modulelist[3](x)

    #     x = self.mid_block1(x, t)
    #     x = self.mid_attn(x)
    #     x = self.mid_block2(x, t)

    #     for modulelist in self.ups:
    #         x = torch.cat((x, h.pop()), dim=1)
    #         x = modulelist[0](x, t)
    #         x = modulelist[1](x, t)
    #         x = modulelist[2](x)
    #         x = modulelist[3](x)

    #     x = self.final_conv(x)
    #     return x

    def forward(self, x, time):
        t = self.time_mlp(time)

        for layer in self.linear_layers:
            x = layer(x, t)
        x = self.final_linear(x)

        return x

