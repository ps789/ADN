import torch
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config

from model import UNet
from config import DiffusionConfig
from diffusion import GaussianDiffusion, make_beta_schedule
import numpy as np

import argparse

@torch.no_grad()
# Generate samples using diffusion model 
def p_sample_loop(self, model, noise, device, noise_fn=torch.randn, capture_every=1000):
    img = noise
    imgs = []

    for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps):
        img = self.p_sample(
            model,
            img,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            noise_fn=noise_fn,
        )

        if i % capture_every == 0:
            imgs.append(img)

    imgs.append(img)
    return imgs

@torch.no_grad()
# Generate samples using ADN model
def generator_sample_loop(self, model, noise, device):
    img = noise
    N, _, _, _ = noise.shape
    for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps):
        img = model(img, torch.tensor([i] * N, device = device))
    return [img]

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='GAN Training')

    # Dataset/Dataloader parameters
    parser.add_argument('--ckpt_root_dir', type=str,
        default='./checkpoint/cifar10/adn/')
    parser.add_argument('--model_type', type=str,
        default='adn_diffusion_')
    parser.add_argument('--file', type=str,
        default='000000')

    return parser.parse_args()
        
if __name__ == "__main__":
    conf = load_config(DiffusionConfig, "config/diffusion_adn.conf", show=False)
    args = _parse_args()
    ckpt = torch.load(f'{args.ckpt_root_dir}{args.model_type}{args.file}.pt')
    
    generator = conf.generator.make()
    generator.load_state_dict(ckpt["generator"])
    generator = generator.to("cuda")
    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to("cuda")
    for i in range(1):
        noise = torch.randn([64, 3, 32, 32], device="cuda")
        imgs = generator_sample_loop(diffusion, generator, noise, "cuda")
        # np.save(f"eval/diffusion_samples_{i}.npy", imgs[-1].detach().cpu().numpy())

    save_image(imgs[0], "sample.png", normalize=True, range=(-1, 1), nrow=8)
