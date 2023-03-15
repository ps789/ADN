import torchvision.utils as vutils
from train import _parse_args
from utils import CIFAR10DataLoader, get_device
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import Generator, Discriminator


def plot_training(args, dataloader):
    import matplotlib.pyplot as plt

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(get_device(args))[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

def plot_generated(args, generator):
    noise = torch.randn(64, args.latent_size, 1, 1, device=get_device(args))
    with torch.no_grad():
        fake = generator(noise).detach().cpu()
    plt.imshow(np.transpose(vutils.make_grid(fake.to(get_device(args))[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig('samples.png', bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    args = _parse_args()
    dataloader = CIFAR10DataLoader(args)

    generator = Generator(args)
    checkpoint = torch.load('../checkpoint/gan/generator_000700.pt')
    generator.load_state_dict(checkpoint['model'])
    generator.to(get_device(args))
    plot_generated(args, generator)