from utils import get_device
import torch
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
import torch.nn as nn
from models import Generator, Discriminator
from utils import get_device, weights_init

class GAN_Wrapper(nn.Module):
    def __init__(self, args, t):
        super(GAN_Wrapper, self).__init__()

        self.args = args
        self.device = get_device(self.args)

        # nc = number of channels
        # ngf = number of generator features
        self.generator = Generator(args, t).to(self.device)
        self.discriminator = Discriminator(args).to(self.device)
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.loss = nn.BCELoss()
        self.t = t

    def forward(self, input):
        return self.generator(input)

    def train_batch(self, args, tgt, diffusion_process):

        real_label = 1.
        fake_label = 0.

        # Format batch
        label = torch.full((args.batch_size,), real_label, dtype=torch.float, device=self.device)

        # Update D
        self.discriminator.zero_grad()
        # Forward pass real batch through D
        if self.t == 0:
            real = tgt
        else:
            real = diffusion_process.q_sample(tgt, torch.tensor([self.t-1] * args.batch_size, device = self.device))
        output = self.discriminator(real).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate fake image batch with G
        fake = self.generator(diffusion_process.q_sample(real, torch.tensor([0] * args.batch_size, device = self.device)))
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = self.discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.loss(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizer_discriminator.step()

        # Update G
        self.generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.loss(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizer_generator.step()

        return errG.item(), errD.item(), D_x, D_G_z1, D_G_z2
    def train_batch_generator(self, args, tgt, diffusion_process):

        real_label = 1.
        fake_label = 0.

        # Format batch
        label = torch.full((args.batch_size,), real_label, dtype=torch.float, device=self.device)

        # Update D
        self.discriminator.zero_grad()
        # Forward pass real batch through D
        if self.t == 0:
            real = tgt
        else:
            real = diffusion_process.q_sample(tgt, torch.tensor([self.t-1] * args.batch_size, device = self.device))
        output = self.discriminator(real).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.loss(output, label)
        # Calculate gradients for D in backward pass
        # errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate fake image batch with G
        fake = self.generator(diffusion_process.q_sample(real, torch.tensor([0] * args.batch_size, device = self.device)))
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = self.discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.loss(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        # errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        # self.optimizer_discriminator.step()

        # Update G
        self.generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.loss(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizer_generator.step()

        return errG.item(), errD.item(), D_x, D_G_z1, D_G_z2
