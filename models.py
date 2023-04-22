from utils import get_device
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, args, t):
        super(Generator, self).__init__()

        self.args = args
        self.device = get_device(self.args)
        self.projection = nn.Linear(args.num_channels * args.image_size ** 2, args.latent_size)

        # nc = number of channels
        # ngf = number of generator features
        module_list = [# input is latent, going into a convolution
        nn.ConvTranspose2d(self.args.latent_size, self.args.generator_features * 4, 4, 1, 0, bias=True),
        nn.BatchNorm2d(self.args.generator_features * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 4 x 4
        nn.ConvTranspose2d(self.args.generator_features * 4, self.args.generator_features * 2, 4, 2, 1, bias=True),
        nn.BatchNorm2d(self.args.generator_features * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 8 x 8
        nn.ConvTranspose2d(self.args.generator_features * 2, self.args.generator_features, 4, 2, 1, bias=True),
        nn.BatchNorm2d(self.args.generator_features),
        nn.ReLU(True),
        # state size. (ngf) x 16 x 16
        nn.ConvTranspose2d(self.args.generator_features, self.args.num_channels, 4, 2, 1, bias=True),
        # state size. (nc) x 32 x 32
        ]
        if t == 0:
            module_list.append(nn.Tanh())
        self.network = nn.Sequential(
            *module_list
        )

    def forward(self, input):
        input = torch.reshape(input, (input.shape[0], -1))
        input = self.projection(input)
        input = input[:, :, None, None]
        return self.network(input)

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.args = args
        self.device = get_device(self.args)

        # nc = number of channels
        # ndf = number of discriminator features
        self.network = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(self.args.num_channels, self.args.discriminator_features, 4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(self.args.discriminator_features, self.args.discriminator_features * 2, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.args.discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(self.args.discriminator_features * 2, self.args.discriminator_features * 4, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.args.discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(self.args.discriminator_features * 4, 1, 4, stride=1, padding=0, bias=True),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.network(input)
