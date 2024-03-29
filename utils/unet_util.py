# adapted from Filip Basara's GitHub repo https://github.com/filipbasara0/simple-diffusion
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

from utils.ch14util import Attention, LayerNorm


def get_downsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
            nn.Conv2d(in_dim * 4, hidden_dim, 1))
    else:
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)


def get_attn_layer(in_dim, is_last):
    if is_last:
        return Residual(PreNorm(in_dim, Attention(in_dim)))
    else:
        return nn.Identity()


def get_upsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.Conv2d(in_dim, hidden_dim, 3, padding=1))
    else:
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)


def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - 1.0)

    emb = torch.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]

    return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 temb_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=8):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(), torch.nn.Linear(temb_channels, out_channels))

        self.residual_conv = nn.Conv2d(
            in_channels, out_channels=out_channels,
            kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)

        self.norm1 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.nonlinearity = nn.SiLU()

    def forward(self, x, temb):
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)

        temb = self.time_emb_proj(self.nonlinearity(temb))
        x += temb[:, :, None, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)

        return x + residual


class UNet(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_dims=[128, 256, 512, 1024],
                 image_size=64):
        super(UNet, self).__init__()

        self.sample_size = image_size
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims

        timestep_input_dim = hidden_dims[0]
        time_embed_dim = timestep_input_dim * 4

        self.time_embedding = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim))

        self.init_conv = nn.Conv2d(in_channels,
                                   out_channels=hidden_dims[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        down_blocks = []

        in_dim = hidden_dims[0]
        for idx, hidden_dim in enumerate(hidden_dims[1:]):
            is_last = idx >= (len(hidden_dims) - 2)
            down_blocks.append(
                nn.ModuleList([
                    ResidualBlock(in_dim, in_dim, time_embed_dim),
                    ResidualBlock(in_dim, in_dim, time_embed_dim),
                    get_attn_layer(in_dim, is_last),
                    get_downsample_layer(in_dim, hidden_dim, is_last)
                ]))
            in_dim = hidden_dim

        self.down_blocks = nn.ModuleList(down_blocks)

        mid_dim = hidden_dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)

        up_blocks = []
        in_dim = mid_dim
        for idx, hidden_dim in enumerate(list(reversed(hidden_dims[:-1]))):
            is_last = idx >= (len(hidden_dims) - 2)
            up_blocks.append(
                nn.ModuleList([
                    ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                    ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                    get_attn_layer(in_dim, is_last),
                    get_upsample_layer(in_dim, hidden_dim, is_last)
                ]))
            in_dim = hidden_dim

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out_block = ResidualBlock(hidden_dims[0] * 2, hidden_dims[0],
                                       time_embed_dim)
        self.conv_out = nn.Conv2d(hidden_dims[0], out_channels=3, kernel_size=1)

    def forward(self, sample, timesteps):
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps],
                                     dtype=torch.long,
                                     device=sample.device)

        timesteps = torch.flatten(timesteps)
        timesteps = timesteps.broadcast_to(sample.shape[0])

        t_emb = sinusoidal_embedding(timesteps, self.hidden_dims[0])
        t_emb = self.time_embedding(t_emb)

        x = self.init_conv(sample)
        r = x.clone()

        skips = []
        for block1, block2, attn, downsample in self.down_blocks:
            x = block1(x, t_emb)
            skips.append(x)

            x = block2(x, t_emb)
            x = attn(x)
            skips.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for block1, block2, attn, upsample in self.up_blocks:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, t_emb)

            x = torch.cat((x, skips.pop()), dim=1)
            x = block2(x, t_emb)
            x = attn(x)

            x = upsample(x)

        x = self.out_block(torch.cat((x, r), dim=1), t_emb)
        out = self.conv_out(x)
        return {"sample": out}
