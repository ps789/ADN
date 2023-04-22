import torch
import torch.nn as nn
from gan_wrapper import GAN_Wrapper

class Chain_GAN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = 'cpu' if args.cpu else 'cuda'

        # GANs
        # t = 0: GAN which takes as input white noise
        # t = T: GAN which outputs the generated image
        self.gans = nn.ModuleList()
        for t in range(args.n_gan):
            self.gans.append(GAN_Wrapper(args, t))

    # Forward pass the entire chain of GANs
    def forward(self, chain_start = 0, chain_end = None, input = None, output_img = False):

        # Default values: run entire chain with random noise input
        if chain_end is None:
            chain_end = self.args.n_gan
        if input is None:
            input = torch.randn(self.args.batch_size, self.args.num_channels, self.args.image_size, self.args.image_size, device = self.device)

        # Iterate through the chain
        for i in range(chain_start, chain_end):
            output = self.gans[chain_end - 1 - i](input)
            input = output
        return input

    # Train a single GAN in the chain
    def train_link(self, tgt, gan_idx, diffusion_process):
        # Train the GAN
        err_G, err_D, _, _, _ = self.gans[gan_idx].train_batch(self.args, tgt, diffusion_process)
        return err_G, err_D

    # Train a single GAN in the chain
    def train_link_generator(self, tgt, gan_idx, diffusion_process):
        # Train the GAN
        err_G, err_D, _, _, _ = self.gans[gan_idx].train_batch_generator(self.args, tgt, diffusion_process)
        return err_G, err_D
