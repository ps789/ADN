from utils import CIFAR10DataLoader, get_device
from get_args import get_args
from gan_wrapper import GAN_Wrapper
from chain_gan import Chain_GAN
import torch
import torchvision
import os

if __name__ == '__main__':

    # Load
    args = get_args()
    dataloader = CIFAR10DataLoader(args)
    device = get_device(args)
    chain_gan = Chain_GAN(args).to(device)
    checkpoint = torch.load(os.path.join(args.checkpoint_path, 'chain_gan_100.pt'))
    chain_gan.load_state_dict(checkpoint['model_state_dict'])

    # Forward pass
    generated_img = chain_gan(output_img = True)
    torchvision.utils.save_image(generated_img, "sample.png", normalize=True, range=(-1, 1), nrow=8)
