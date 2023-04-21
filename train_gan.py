from utils import CIFAR10DataLoader, get_device
from get_args import get_args
from diffusion import GaussianDiffusion, make_beta_schedule
from chain_gan import Chain_GAN
from tqdm import tqdm
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load model
    args = get_args()
    dataloader = CIFAR10DataLoader(args)
    chain_gan = Chain_GAN(args)
    device = get_device(args)
    chain_gan.to(device)

    # Load diffusion process
    betas = make_beta_schedule(args.beta_schedule, args.n_gan)
    diffusion = GaussianDiffusion(betas)
    diffusion.to(device)
    T = args.n_gan - 1

    # Training loop
    for epoch in range(args.num_epochs + 1):

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
            for t in list(range(T + 1))[::-1]:
                err_G, err_D = chain_gan.train_link(tgt = data_batch, gan_idx = t, diffusion_process = diffusion)
            # plt.show()

            # Output values
            epoch_errG += err_G
            epoch_errD += err_D
            pbar.set_description(
                'Epoch:{epoch}, Loss_D:{errD:.2f}, Loss_G:{errG:.2f}'.format(epoch = epoch, errD = epoch_errD / (i + 1), errG = epoch_errG / (i + 1))
            )

        # Save model
        if epoch % args.save_model_frequency == 0:
            checkpoint = {
                'model_state_dict': chain_gan.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'chain_gan_{epoch}.pt'.format(epoch = epoch)))
