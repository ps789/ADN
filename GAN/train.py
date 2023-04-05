import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm

from utils import CIFAR10DataLoader, get_device, weights_init
from models import Generator, Discriminator

def tqdm_write(string):
    tqdm.write(string, end='')

def train(args, dataloader):
    device = get_device(args)

    generator = Generator(args).to(device)
    discriminator = Discriminator(args).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    loss = nn.BCELoss()
    fixed_noise = torch.randn(64, args.latent_size, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.
    optimizer_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting training...")
    for epoch in range(args.num_epochs):
        pbar = tqdm(enumerate(dataloader))
        for i, data in pbar:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = loss(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.latent_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = loss(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizer_discriminator.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_generator.step()

            # Output training stats
            if i % 20 == 0:
                pbar.set_description('[%3d/%d][%3d/%d]    Loss_D: %.4f    Loss_G: %.4f    D(x): %.4f    D(G(z)): %.4f / %.4f'
                      % (epoch, args.num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        if epoch % args.save_model_frequency == 0:
            torch.save(
                {
                    "model": generator.state_dict(),
                    "optimizer": optimizer_generator.state_dict(),
                },
                f"{args.checkpoint_location}/generator_{str(epoch).zfill(6)}.pt",
            )

            torch.save(
                {
                    "model": discriminator.state_dict(),
                    "optimizer": optimizer_discriminator.state_dict(),
                },
                f"{args.checkpoint_location}/discriminator_{str(epoch).zfill(6)}.pt",
            )


def plot(args, dataloader):
    import matplotlib.pyplot as plt
    import numpy as np
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(get_device(args))[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


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
    train(args, dataloader)
