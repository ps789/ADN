import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

def get_device(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cpu:
        device = torch.device('cpu')
    return device

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def CIFAR10DataLoader(args):
	dataset = CIFAR10(root=args.dataset_root, train=True, download=True,
			 transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

	return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers = True, drop_last = True)