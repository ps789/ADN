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

    # Create src_diffusion_probability schedule
    src_diffusion_prob_schedule = torch.linspace(args.src_diffusion_prob_start, args.src_diffusion_prob_end, args.num_epochs + 1)

    # Training loop
    for epoch in range(args.num_epochs + 1):

        epoch_errG = 0
        epoch_errD = 0

        # For each batch
        pbar = tqdm(enumerate(dataloader))
        for i, data in pbar:

            data_batch = data[0].to(device)

            # Get the noisiest source image from forward diffusion process
            noise = diffusion.q_sample(data_batch, torch.tensor([T] * args.batch_size, device = device))

            # Train each GAN in the chain
            # t = 0: Original image
            # t = T: Noise
            # fig, ax = plt.subplots(2, T + 1)
            for t in list(range(T + 1))[::-1]:

                # If we are at the last GAN, target is original image. Else use diffusion
                if t == 0:
                    tgt = data_batch
                else:
                    tgt = diffusion.q_sample(data_batch, torch.tensor([t - 1] * args.batch_size, device = device))

                # Train chain up to time t
                src = None
                src_diffusion_prob = src_diffusion_prob_schedule[epoch]
                if torch.rand(1) < src_diffusion_prob:
                    src = diffusion.q_sample(data_batch, torch.tensor([t] * args.batch_size, device = device))

                # DEBUG: Show noisy images
                # img = tgt[0]
                # img = img.cpu().detach().numpy()
                # img = np.transpose((img + 1) / 2, (1, 2, 0))
                # ax[1, t].imshow(img)
                # ax[1, t].set_xlabel(str(t) + ' tgt')
                # img = src[0]
                # img = img.cpu().detach().numpy()
                # img = np.transpose((img + 1) / 2, (1, 2, 0))
                # ax[0, t].imshow(img)
                # ax[0, t].set_xlabel(str(t) + ' src')
                # continue

                err_G, err_D = chain_gan.train_link(noise = noise, tgt = tgt, gan_idx = args.n_gan - t - 1, src = src)
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
