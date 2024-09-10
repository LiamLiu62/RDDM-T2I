import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm
from datasets import load_dataset

from diffusers import AutoencoderKL
import clip
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])


# helpers functions

def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    """
    Convert a number to groups.
    i.e. num_to_groups(5, 2) -> [2, 2, 1]
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions

def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x ** ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y / y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x ** ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y / y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3 + mu, 3 + mu, timesteps, dtype=np.float32)
        y = np.e ** (-((x - mu) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * (sigma ** 2))
        y = torch.from_numpy(y)
        alphas = y / y.sum()
    else:
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float32)
    assert (alphas.sum() - 1).abs() < 1e-6

    return alphas * sum_scale


# autoencoderkl class

class AutoencoderklFeatureExtractor(object):
    def __init__(self, model_name='CompVis/stable-diffusion-v1-4'):
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')

    def encode_img(self, img):  # img_size: [256, 256]
        device = img.device
        self.vae.to(device)
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        latents = self.vae.encode(normalize_to_neg_one_to_one(img))
        return 0.18215 * latents.latent_dist.sample()

    def decode_latent(self, latents):  # latents_size: [32, 32]
        device = latents.device
        self.vae.to(device)
        latents = (1 / 0.18215) * latents
        img = self.vae.decode(latents).sample
        img = unnormalize_to_zero_to_one(img).clamp(0, 1)
        return img.detach()

    def reconstruct_img(self, img):
        latents = self.encode_img(img)
        return self.decode_latent(latents)


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds

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


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class ConvUpsample(nn.Module):
    def __init__(self):
        super(ConvUpsample, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=1, padding=0),  # (256, 4, 4)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 8)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),  # (4, 32, 32)
        )

    def forward(self, x):
        return self.upsample(x)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=4,  # here use latent space features
            self_condition=False,
            resnet_block_groups=8
    ):

        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * 2 + channels * (1 if self_condition else 0)  # x_T and x_input

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        # I_T and I_in, x=[B, 8, 32, 32]
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class UnetRes(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=4,
            self_condition=False,
            resnet_block_groups=8,
    ):
        super().__init__()
        self.channels = channels
        self.out_dim = default(out_dim, channels)
        self.self_condition = self_condition
        # determine dimensions
        self.unet0 = Unet(dim,
                          init_dim=init_dim,
                          out_dim=out_dim,
                          dim_mults=dim_mults,
                          channels=channels,
                          self_condition=self_condition,
                          resnet_block_groups=resnet_block_groups)

        self.unet1 = Unet(dim,
                          init_dim=init_dim,
                          out_dim=out_dim,
                          dim_mults=dim_mults,
                          channels=channels,
                          self_condition=self_condition,
                          resnet_block_groups=resnet_block_groups)

    def forward(self, x, time, x_self_cond=None):
        # I_T and I_in, x_in=[B, 8, 32, 32]
        return self.unet0(x, time[0], x_self_cond=x_self_cond), self.unet1(x, time[1], x_self_cond=x_self_cond)


# residual diffusion trainer class

class ResidualDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_res_noise',
            ddim_sampling_eta=0.,
            sum_scale=None,
    ):
        super().__init__()
        assert not (
                type(self) == ResidualDiffusion and model.channels != model.out_dim)
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective

        self.sum_scale = 1.
        # if self.condition:
        #     self.sum_scale = sum_scale if sum_scale else 0.01
        #     ddim_sampling_eta = 0.
        # else:
        #     self.sum_scale = sum_scale if sum_scale else 1.

        alphas = gen_coefficients(timesteps, schedule="decreased")
        betas2 = gen_coefficients(timesteps, schedule="increased", sum_scale=self.sum_scale)

        alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
        betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1 - alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1', betas2_cumsum_prev / betas2_cumsum)
        register_buffer('posterior_mean_coef2',
                        (betas2 * alphas_cumsum_prev - betas2_cumsum_prev * alphas) / betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2 / betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

        # VAE related parameters
        self.vae = AutoencoderklFeatureExtractor()

        # DallE2 related parameters
        self.prior_config = TrainDiffusionPriorConfig.from_json_path(
            "/root/autodl-tmp/RDDM-t2i/prior_weights/prior_config.json").prior
        self.prior = self.prior_config.create()
        self.prior_model_state = torch.load("/root/autodl-tmp/RDDM-t2i/prior_weights/prior_latest.pth")
        self.prior.load_state_dict(self.prior_model_state, strict=True)

        # Upsample related parameters
        self.upsample = ConvUpsample()

    def init(self):
        timesteps = 1000
        alphas = gen_coefficients(timesteps, schedule="average", ratio=1)  # different
        betas2 = gen_coefficients(timesteps, schedule="increased", sum_scale=self.sum_scale, ratio=3)  # different

        alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
        betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])  # different
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])  # different

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.one_minus_alphas_cumsum = 1 - alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev / betas2_cumsum
        self.posterior_mean_coef2 = (betas2 * alphas_cumsum_prev - betas2_cumsum_prev * alphas) / betas2_cumsum
        self.posterior_mean_coef3 = betas2 / betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
                (x_t - x_input - (extract(self.alphas_cumsum, t, x_t.shape) - 1)
                 * pred_res) / extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
                (x_t - extract(self.alphas_cumsum, t, x_t.shape) * x_input -
                 extract(self.betas_cumsum, t, x_t.shape) * noise) / extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
                x_t - extract(self.alphas_cumsum, t, x_t.shape) * x_res -
                extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t - extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape) / extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
                extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_self_cond=None, clip_denoised=True):
        x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in, [t, t], x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_denoised else identity

        # predict I_0 using I_in and epsilon
        pred_res = model_output[0]
        pred_noise = model_output[1]
        pred_res = maybe_clip(pred_res)
        x_start = self.predict_start_from_res_noise(x, t, pred_res, pred_noise)
        x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_self_cond=None):
        preds = self.model_predictions(x_input, x, t, x_self_cond)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_input, x=x, t=batched_times,
                                                                          x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        x_input = x_input[0]
        batch, device = shape[0], self.betas.device

        img = x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
        input_add_noise = img

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(x_input, img, t, self_cond)

            if not last:
                img_list.append(img)

        if not last:
            img_list = [input_add_noise] + img_list  # contain all steps
        else:
            img_list = [input_add_noise, img]  # contain the first and the last
        return img_list

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True):
        # simutaneously remove noise and residual

        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        img = x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
        input_add_noise = img

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        eta = 0

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(x_input, img, time_cond, self_cond)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum - alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum - betas2_cumsum_next
            # betas2 = 1-(1-betas2_cumsum)/(1-betas2_cumsum_next)
            betas_cumsum = self.betas_cumsum[time]
            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                                                                                    betas2_cumsum_next - sigma2).sqrt() / betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            if type == "use_pred_noise":  # formula (41)
                img = img - alpha * pred_res - (
                            betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()) * pred_noise + sigma2.sqrt() * noise
            elif type == "use_x_start":  # formula (40)
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum * img + \
                      (1 - sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * x_start + \
                      (
                                  alpha_cumsum_next - alpha_cumsum * sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * pred_res + \
                      sigma2.sqrt() * noise

            if not last:
                img_list.append(img)

        if not last:
            img_list = [input_add_noise] + img_list
        else:
            img_list = [input_add_noise, img]
        return img_list

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True):
        # x_input: [[B, 3, 256, 256]]，列表中只装一个元素
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        with torch.no_grad():
            prior_emb = [self.prior.sample(i) for i in torch.unbind(x_input[0], dim=0)]
            prior_emb = torch.stack(prior_emb, dim=0)  # [B, 1, 768]
        prior_emb = prior_emb.permute(0, 2, 1)  # [B, 768, 1]
        prior_emb = prior_emb.unsqueeze(-1)  # [B, 768, 1, 1]

        # Upsample block
        with torch.no_grad():
            prior_emb = self.upsample(prior_emb)  # prior: [B, 4, 32, 32]

        batch_size, channels, h, w = prior_emb.shape
        prior_emb = [prior_emb]
        size = (batch_size, channels, h, w)

        return sample_fn(prior_emb, size, last=last)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                x_start + extract(self.alphas_cumsum, t, x_start.shape) * x_res +
                extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, img_text, t, noise=None):
        x_input = img_text[1]  # I_in: prior, [B, 4, 32, 32]
        x_start = img_text[0]  # I_0: gt latent, [B, 4, 32, 32]

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start  # I_res = I_in - I_0

        b, c, h, w = x_start.shape

        # noise sample, use noise and residual to get x_T
        x = self.q_sample(x_start, x_res, t, noise=noise)  # I_T, [B, 4, 32, 32]

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x_input, x, t, 0).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        x_in = torch.cat((x, x_input), dim=1)  # I_T and I_in, x_in=[B, 8, 32, 32]
        model_out = self.model(x_in, [t, t], x_self_cond)

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000 * self.loss_fn(x_u, u_gt, reduction='none')
            return [loss]
        else:
            loss_list = []
            for i in range(len(model_out)):
                loss = self.loss_fn(model_out[i], target[i], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)
            return loss_list

    def forward(self, img_text, *args, **kwargs):
        print('img_text:', img_text)

        # img_text[0]: [B, 3, 256, 256], img_text[1]: [B, 1, 77]
        assert isinstance(img_text, list)
        b, c, h, w, device, img_size, = *img_text[0].shape, img_text[0].device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # VAE latent space
        img = img_text[0]  # [B, 3, 256, 256]
        with torch.no_grad():
            latent = self.vae.encode_img(img)  # latent: [B, 4, 32, 32]

        # DallE2 prior
        text = img_text[1]  # [B, 1, 77]
        with torch.no_grad():
            prior_emb = [self.prior.sample(i) for i in torch.unbind(text, dim=0)]
            prior_emb = torch.stack(prior_emb, dim=0)  # [B, 1, 768]
        prior_emb = prior_emb.permute(0, 2, 1)  # [B, 768, 1]
        prior_emb = prior_emb.unsqueeze(-1)  # [B, 768, 1, 1]

        # Upsample block
        prior_emb = self.upsample(prior_emb)  # prior: [B, 4, 32, 32]

        img_text = [latent, prior_emb]  # img_text: [[B, 4, 32, 32], [B, 4, 32, 32]]

        return self.p_losses(img_text, t, *args, **kwargs)


# text-image dataset class
class TextImageDataset(Dataset):
    def __init__(self, folder, image_size=256):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.dataset = load_dataset("imagefolder", data_dir=folder, split='train')

        # Here use medical image, so do not use flip augmentation
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

        # tokenize the text
        self.clip_model, self.preprocess = clip.load("ViT-L/14")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image'].convert('RGB')
        img = self.transform(img)

        # get tokenized text tokens
        text = self.dataset[idx]['text']
        if not text:
            text = 'no description'
        tokenized_text = clip.tokenize(text, context_length=77, truncate=True)

        # img: [3, 256, 256], tokenized_text: [1, 77]
        return img, tokenized_text


# trainer class

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=False,
            sub_dir=False,
            crop_patch=False,
            generation=False,
            num_unet=2
    ):
        super().__init__()

        self.model = diffusion_model
        self.folder = folder

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.save_every = save_every

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'

        self.num_samples = num_samples
        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision='fp16' if fp16 else 'no')
        self.accelerator.native_amp = amp
        self.sub_dir = sub_dir
        # self.crop_patch = crop_patch
        self.latent_image_size = diffusion_model.image_size  # 32
        self.image_size = 8 * diffusion_model.image_size  # 256

        # load image-text dataset
        self.dataset = TextImageDataset(folder, self.image_size)

        dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.dl = cycle(self.accelerator.prepare(dl))

        # optimizer
        self.opt0 = Adam(diffusion_model.model.unet0.parameters(), lr=train_lr, betas=adam_betas)
        self.opt1 = Adam(diffusion_model.model.unet1.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.set_results_folder(results_folder)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt0, self.opt1 = self.accelerator.prepare(self.model, self.opt0, self.opt1)
        self.device = self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt0': self.opt0.state_dict(),
            'opt1': self.opt1.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, self.results_folder / f'model-{milestone}.pt')

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')

        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])

            self.step = data['step']
            self.opt0.load_state_dict(data['opt0'])
            self.opt1.load_state_dict(data['opt1'])
            self.ema.load_state_dict(data['ema'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = [0, 0]
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data = [item.to(self.device) for item in data]

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        for i in range(2):
                            loss[i] = loss[i] / self.gradient_accumulate_every
                            total_loss[i] += total_loss[i] + loss[i].item()

                    union_loss = loss[0] + loss[1]
                    self.accelerator.backward(union_loss)
                    # self.accelerator.backward(loss[0])
                    # self.accelerator.backward(loss[1])

                accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                accelerator.wait_for_everyone()

                self.opt0.step()
                self.opt1.step()
                self.opt0.zero_grad()
                self.opt1.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_every == 0:
                        milestone = self.step // self.save_every

                        self.sample(milestone, last=True)

                        self.save(milestone)

                pbar.set_description(f'loss_unet0: {total_loss[0]:.4f}, loss_unet1: {total_loss[1]:.4f}')
                pbar.update(1)

        accelerator.print('training complete')

    def sample(self, milestone, last=True):
        self.ema.ema_model.eval()

        with torch.no_grad():
            batches = self.num_samples
            data = next(self.dl)
            x_input_sample = [data[1].to(self.device)]

            all_images_list = list(self.ema.ema_model.sample(x_input_sample, batch_size=batches, last=last))
            # all_images_list[0]: [2, 4, 32, 32], all_images_list[1]: [2, 4, 32, 32]
            all_images = torch.cat(all_images_list, dim=0)                              # [4, 4, 32, 32]

            # decode from latent space into image
        with torch.no_grad():
            reconstructed_images = self.model.vae.decode_latent(all_images)         # [4, 3, 256, 256]
        nrow = 2
        file_name = f'sample-{milestone}.png'
        utils.save_image(reconstructed_images, str(self.results_folder / file_name), nrow=nrow)
        print("sampe-save " + file_name)


    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)





