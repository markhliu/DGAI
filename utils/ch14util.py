# my_utils.py
import os
from PIL import Image
import torch
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from torchvision.transforms import (CenterCrop,
    Compose,InterpolationMode,RandomHorizontalFlip,
    Resize,ToTensor)

resolution=64

augmentations = Compose([
    Resize(resolution, 
    interpolation=InterpolationMode.BILINEAR),
    CenterCrop(resolution),
    RandomHorizontalFlip(),
    ToTensor(),])

def transforms(examples):
    images = [augmentations(image.convert("RGB"))
        for image in examples["image"]]
    return {"input": images}


class CustomDataset(Dataset):

    def __init__(self, data_df, transforms):
        image_paths = []
        for idx, row in data_df.iterrows():
            image_path = row["image_path"]

            image_paths.append(image_path)
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return {"input": image}


def plot_losses(losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.plot(losses, label='train')
    plt.legend()
    plt.savefig(f"{out_dir}/losses.png")
    plt.clf()


def save_images(generated_images, epoch, batch_size):
    images = generated_images["sample"]
    images_processed = (images * 255).round().astype("uint8")
    out_dir = f"./{epoch}/"
    os.makedirs(out_dir)
    for idx, image in enumerate(images_processed):
        image = Image.fromarray(image)
        image.save(f"{out_dir}/{epoch}_{idx}.jpeg")
    utils.save_image(generated_images["sample_pt"],
                        f"{out_dir}/{epoch}_grid.jpeg",
                        nrow=batch_size // 4)


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()

    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)

    return values


def clip(tensor, min_value=None, max_value=None):
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    elif isinstance(tensor, torch.Tensor):
        return torch.clamp(tensor, min_value, max_value)

    raise ValueError("Tensor format is not valid is not valid - " \
        f"should be numpy array or torch tensor. Got {type(tensor)}.")
        
        
        
        
        
        
# attention related 
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)       
        
        
        

# adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
# adapted from https://github.com/filipbasara0/simple-diffusion
import math
from typing import Union

import numpy as np
import torch
from tqdm import tqdm



def cosine_beta_schedule(timesteps, beta_start=0.0, beta_end=0.999, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)


class DDIMScheduler:

    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="cosine",
        clip_sample=True,
        set_alpha_to_one=True
    ):

        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(num_train_timesteps,
                                              beta_start=beta_start,
                                              beta_end=beta_end)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}")

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.final_alpha_cumprod = np.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps, offset=0):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(0, 1000, 1000 // num_inference_steps)[::-1].copy()
        self.timesteps += offset

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        eta: float = 1.0,
        use_clipped_model_output: bool = True,
        generator=None,
    ):

        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t**(0.5) * model_output) / alpha_prod_t**(0.5)

        if self.clip_sample:
            pred_original_sample = clip(pred_original_sample, -1, 1)


        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**(0.5)

        if use_clipped_model_output:
            model_output = (sample -
                            alpha_prod_t**(0.5) * pred_original_sample) / beta_prod_t**(0.5)

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2)**(0.5) * model_output

        prev_sample = alpha_prod_t_prev**(0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            device = model_output.device if torch.is_tensor(model_output) else "cpu"
            noise = torch.randn(model_output.shape, generator=generator).to(device)
            variance = self._get_variance(timestep, prev_timestep)**(0.5) * eta * noise

            if not torch.is_tensor(model_output):
                variance = variance.numpy()

            prev_sample = prev_sample + variance

        return prev_sample

    def add_noise(self, original_samples, noise, timesteps):
        timesteps = timesteps.cpu()
        sqrt_alpha_prod = self.alphas_cumprod[timesteps]**0.5
        sqrt_alpha_prod = match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps])**0.5
        sqrt_one_minus_alpha_prod = match_shape(sqrt_one_minus_alpha_prod, original_samples)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    @torch.no_grad()
    def generate(self,model,device,batch_size=1,generator=None,
         eta=1.0,use_clipped_model_output=True,num_inference_steps=50):
        imgs=[]
        image=torch.randn((batch_size,model.in_channels,model.sample_size,
            model.sample_size),generator=generator).to(device)
        self.set_timesteps(num_inference_steps)
        for t in tqdm(self.timesteps):
            model_output = model(image, t)["sample"]
            image = self.step(model_output,t,image,eta,
                  use_clipped_model_output=use_clipped_model_output)
            img = unnormalize_to_zero_to_one(image)
            img = img.cpu().permute(0, 2, 3, 1).numpy()
            imgs.append(img)
        image = unnormalize_to_zero_to_one(image)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return {"sample": image}, imgs
    def __len__(self):
        return self.num_train_timesteps
        
        
        
        
        
        
        
        
        
