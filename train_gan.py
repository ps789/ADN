from utils import CIFAR10DataLoader, get_device
from get_args import get_args
from diffusion import GaussianDiffusion, make_beta_schedule
from chain_gan import Chain_GAN
from tqdm import tqdm
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import random
from ema import ModelEMA

def accumulate(model1, model2, decay=0.9999):
    # mod1 = dict(model1.named_modules())
    # mod2 = dict(model2.named_modules())
    # prams = dict(model1.named_parameters())
    # for km in mod1.keys():
    #     par1 = dict(mod1[km].named_parameters())
    #     par2 = dict(mod2[km].named_parameters())
    #     for k in par1.keys():
    #         par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
    buf1 = dict(model1.named_buffers())
    buf2 = dict(model2.named_buffers())
    for k in buf1.keys():
        if "num_batches_tracked" not in k:
            buf1[k].data.mul_(decay).add_(buf2[k].data, alpha=1 - decay)

if __name__ == '__main__':

    # Load model
    args = get_args()
    dataloader = CIFAR10DataLoader(args)
    chain_gan = Chain_GAN(args)
    ema = Chain_GAN(args)
    device = get_device(args)
    chain_gan.to(device)
    ema = ModelEMA(chain_gan, device = device)
    # ckpt = torch.load("./checkpoint/cifar10/chain_gan/chain_gan_100.pt")
    # chain_gan.load_state_dict(ckpt["model_state_dict"])
    # ema.ema.load_state_dict(ckpt["ema_state_dict"])
    # ema.update(chain_gan, 0)
    # ema.load_state_dict(ckpt["ema_state_dict"])

    # Load diffusion process
    betas = make_beta_schedule(args.beta_schedule, args.n_gan)
    diffusion = GaussianDiffusion(betas)
    diffusion.to(device)
    T = args.n_gan - 1

    # Training loop
    for epoch in range(1, args.num_epochs + 1):

        epoch_errG = 0
        epoch_errD = 0

        # For each batch
        pbar = tqdm(enumerate(dataloader))
        for i, data in pbar:

            data_batch = data[0].to(device)
            # Train each GAN in the chain
            # t = 0: Original image
            # t = T: Noise
            # fig, ax = plt.subplots(2, T + 1)
            #for t in list(range(T + 1)):
            t = random.randint(0, T)
            if i % 5 == 0:
                err_G, err_D = chain_gan.train_link(tgt = data_batch, gan_idx = t, diffusion_process = diffusion)
            else:
                err_G, err_D = chain_gan.train_link_generator(tgt = data_batch, gan_idx = t, diffusion_process = diffusion)
            # plt.show()

            # Output values
            epoch_errG += err_G
            epoch_errD += err_D
            pbar.set_description(
                'Epoch:{epoch}, Loss_D:{errD:.2f}, Loss_G:{errG:.2f}'.format(epoch = epoch, errD = epoch_errD / (i + 1), errG = epoch_errG / (i + 1))
            )
        ema.update(chain_gan, 0 if epoch < 50 else 0.999)
        generated_img = chain_gan(output_img = True)
        torchvision.utils.save_image(generated_img, f"./sample_images/sample_{epoch}.png", normalize=True, range=(-1, 1), nrow=8)
        generated_img = ema.ema(output_img = True)
        torchvision.utils.save_image(generated_img, f"./sample_images/ema_{epoch}.png", normalize=True, range=(-1, 1), nrow=8)

        diffusion_sample = diffusion.q_sample(data_batch, torch.tensor([0] * args.batch_size, device = device))
        torchvision.utils.save_image(diffusion_sample, "./sample_images/diffusion0.png", normalize=True, range=(-1, 1), nrow=8)

        generated_img = chain_gan(0, 1, diffusion_sample, output_img = True)
        torchvision.utils.save_image(generated_img, f"./sample_images/sample1_{epoch}.png", normalize=True, range=(-1, 1), nrow=8)
        generated_img = ema.ema(0, 1, diffusion_sample, output_img = True)
        torchvision.utils.save_image(generated_img, f"./sample_images/ema1_{epoch}.png", normalize=True, range=(-1, 1), nrow=8)

        diffusion_sample = diffusion.q_sample(diffusion_sample, torch.tensor([0] * args.batch_size, device = device))
        torchvision.utils.save_image(diffusion_sample, "./sample_images/diffusion1_alt.png", normalize=True, range=(-1, 1), nrow=8)

        diffusion_sample = diffusion.q_sample(data_batch, torch.tensor([1] * args.batch_size, device = device))
        torchvision.utils.save_image(diffusion_sample, "./sample_images/diffusion1.png", normalize=True, range=(-1, 1), nrow=8)

        generated_img = chain_gan(0, 2, diffusion_sample, output_img = True)
        torchvision.utils.save_image(generated_img, f"./sample_images/sample2_{epoch}.png", normalize=True, range=(-1, 1), nrow=8)
        generated_img = ema.ema(0, 2, diffusion_sample, output_img = True)
        torchvision.utils.save_image(generated_img, f"./sample_images/ema2_{epoch}.png", normalize=True, range=(-1, 1), nrow=8)
        # Save model
        if epoch % args.save_model_frequency == 0:
            checkpoint = {
                'model_state_dict': chain_gan.state_dict(),
                'ema_state_dict': ema.ema.state_dict()
            }


            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'chain_gan_{epoch}.pt'.format(epoch = epoch)))
