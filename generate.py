import torch
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config

from model import UNet
from config import DiffusionConfig
from diffusion import GaussianDiffusion, make_beta_schedule
import numpy as np

@torch.no_grad()
def p_sample_loop(self, model, noise, class_num, device, noise_fn=torch.randn, capture_every=1000):
    img = noise
    imgs = []

    for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps):
        labels = class_num * torch.ones(img.shape[0], dtype = int, device = 'cuda')
        img = self.p_sample(model, img, labels, 
                            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
                            noise_fn=noise_fn)

        if i % capture_every == 0:
            imgs.append(img)

    imgs.append(img)
    return imgs

if __name__ == "__main__":
    conf = load_config(DiffusionConfig, "config/conditional_diffusion.conf", show=False)
    ckpt = torch.load("checkpoint/diffusion_004000.pt")
    model = conf.model.make()
    model.load_state_dict(ckpt["ema"])
    model = model.to("cuda")
    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to("cuda")

    class_num = 1
    large_batch = False
    
    if large_batch:
        for i in range(50):
            noise = torch.randn([200, 3, 32, 32], device="cuda")
            imgs = p_sample_loop(diffusion, model, noise, class_num, "cuda", capture_every=1000)
            np.save(f"eval/diffusion_samples_{i}.npy", imgs[-1].detach().cpu().numpy())
    else:
        noise = torch.randn([16, 3, 32, 32], device="cuda")
        imgs = p_sample_loop(diffusion, model, noise, class_num, "cuda", capture_every=1000)
        save_image(imgs[-1], "sample.png", normalize=False, range_value=(-1, 1), nrow=4)
