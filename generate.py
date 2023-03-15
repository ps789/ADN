import torch
from tqdm import tqdm
from torchvision.utils import save_image
from tensorfn import load_config
from config import DiffusionConfig
from diffusion import GaussianDiffusion, make_beta_schedule
import numpy as np
import pickle

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

    model_names = ['conditional_diffusion_{0}'.format(str(x).zfill(6)) for x in np.arange(11) * 50000]

    for model_name in model_names:

        conf = load_config(DiffusionConfig, "config/conditional_diffusion.conf", show=False)
        ckpt = torch.load("checkpoint/conditional_diffusion/{0}.pt".format(model_name))
        model = conf.model.make()
        model.load_state_dict(ckpt["ema"])
        model = model.to("cuda")
        betas = conf.diffusion.beta_schedule.make()
        diffusion = GaussianDiffusion(betas).to("cuda")
        
        # Generate images
        num_classes = 10
        num_batches = 2
        batch_size = 500
        all_imgs = {}
        for c in range(num_classes):

            all_imgs[c] = np.zeros((num_batches * batch_size, 3, 32, 32))
            for i in range(num_batches):
                noise = torch.randn([batch_size, 3, 32, 32], device="cuda")
                imgs = p_sample_loop(diffusion, model, noise, c, "cuda", capture_every=1000)
                all_imgs[c][i * batch_size : (i + 1) * batch_size, ...] = imgs[-1].detach().cpu().numpy()

        # Save images
        with open('eval/conditional_diffusion/imgs_{0}.pickle'.format(model_name), 'wb') as p:
            pickle.dump(all_imgs, p)

    # Generate a grid of images:
    # class_num = 0
    # noise = torch.randn([64, 3, 32, 32], device="cuda")
    # imgs = p_sample_loop(diffusion, model, noise, class_num, "cuda", capture_every=1000)
    # save_image(imgs[-1], "sample.png", normalize=False, range_value=(-1, 1), nrow=8)
