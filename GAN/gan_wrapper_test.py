import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm

from utils import CIFAR10DataLoader, get_device, weights_init
from gan_wrapper import GAN_Wrapper
def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='GAN Training')

    # Dataset/Dataloader parameters
    parser.add_argument('--dataset_root', type=str,
        default='../cifar10')
    parser.add_argument('--image_size', type=int,
        default=32)
    parser.add_argument('--batch_size', type=int,
        default=128)
    parser.add_argument('--num_workers', type=int,
        default=1)

    # GAN parameters
    parser.add_argument('--num_channels', type=int,
        default=3)
    parser.add_argument('--latent_size', type=int,
        default=256)
    parser.add_argument('--generator_features', type=int,
        default=64)
    parser.add_argument('--discriminator_features', type=int,
        default=64)

    # Training parameters
    parser.add_argument('--num_epochs', type=int,
        default=100)
    parser.add_argument('--lr', type=float,
        default=0.001)
    parser.add_argument('--beta1', type=float,
        default=0.5)
    parser.add_argument('--beta2', type=float,
        default=0.999)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save_model_frequency', type=int,
        default=2)
    parser.add_argument('--checkpoint_location', type=str,
        default='../checkpoint/')

    return parser.parse_args()



if __name__ == '__main__':
    args = _parse_args()
    dataloader = CIFAR10DataLoader(args)
    gan_wrapper = GAN_Wrapper(args)
    device = get_device(args)
    for epoch in range(args.num_epochs):
        pbar = tqdm(enumerate(dataloader))
        for i, data in pbar:
            data_batch = data[0].to(device)
            errG, errD, D_x, D_G_z1, D_G_z2, _ = gan_wrapper.train_batch(args, data_batch, False)
            if i % 20 == 0:
                pbar.set_description('[%3d/%d][%3d/%d]    Loss_D: %.4f    Loss_G: %.4f    D(x): %.4f    D(G(z)): %.4f / %.4f'
                      % (epoch, args.num_epochs, i, len(dataloader),
                         errD, errG, D_x, D_G_z1, D_G_z2))
