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
        for _ in range(args.n_gan):
            self.gans.append(GAN_Wrapper(args))

    # Forward pass the entire chain of GANs
    def forward(self, chain_start = 0, chain_end = None, input = None, output_img = False):
        
        # Default values: run entire chain with random noise input
        if chain_end is None:
            chain_end = self.args.n_gan
        if input is None:
            input = torch.randn(self.args.batch_size, self.args.num_channels, self.args.image_size, self.args.image_size, device = self.device)

        # Iterate through the chain
        for i in range(chain_start, chain_end):
            output = self.gans[i](input)
            input = output
        return input
    
    # Train a single GAN in the chain
    def train_link(self, noise, tgt, gan_idx, src = None):

        # Forward pass up until gan_idx if src is not provided
        if src is None:
            src = self.forward(chain_start = 0, chain_end = gan_idx, input = noise)
        src.detach()
        
        # Train the GAN
        err_G, err_D, _, _, _ = self.gans[gan_idx].train_batch(self.args, tgt, src)
        return err_G, err_D

    
