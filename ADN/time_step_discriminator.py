from GAN.utils import get_device

import torch.nn as nn
import torch
class ConditionalTimeStepDiscriminator(nn.Module):
    def __init__(self, args):
        super(ConditionalTimeStepDiscriminator, self).__init__()

        self.args = args
        self.device = get_device(self.args)

        # nc = number of channels
        # ndf = number of discriminator features
        self.network = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(self.args.num_channels*2 + 1, self.args.discriminator_features, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(self.args.discriminator_features, self.args.discriminator_features * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.args.discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(self.args.discriminator_features * 2, self.args.discriminator_features * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.args.discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(self.args.discriminator_features * 4, 1, 4, stride=1, padding=0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.network(input)

class ConditionalTimeStepDiscriminator_DualHead(nn.Module):
    def __init__(self, args):
        super(ConditionalTimeStepDiscriminator_DualHead, self).__init__()

        self.args = args
        self.device = get_device(self.args)

        # nc = number of channels
        # ndf = number of discriminator features
        self.network_head = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(self.args.num_channels + 1, self.args.discriminator_features, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(self.args.discriminator_features, self.args.discriminator_features * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.args.discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(self.args.discriminator_features * 2, self.args.discriminator_features * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.args.discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network = nn.Sequential(
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(self.args.discriminator_features * 4 *2, 1, 4, stride=1, padding=0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, conditional, denoised):
        return self.network(torch.cat((self.network_head(conditional), self.network_head(denoised)), dim = 1))
