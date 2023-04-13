import torch
from model import SimpleGenerator
import torchvision
import matplotlib.pyplot as plt
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')

features = torch.randn([128, 3, 32, 32])
train_set = torchvision.datasets.CIFAR10(root='./cifar10', train = True, download = True,
                                            transform = torchvision.transforms.ToTensor())

print(len(train_set))
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = False)
labels = next(iter(train_loader))[0].to('cuda')

generator = SimpleGenerator()
optim = torch.optim.Adam(generator.parameters(), lr = 0.001)
loss_fn = torch.nn.MSELoss()

for i in range(1000):
    optim.zero_grad()
    out = generator(labels, 0)
    loss = loss_fn(out, labels)
    loss.backward()
    optim.step()
    print(loss.item())

for i in range(128):
    img = out[i, ...]
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.figure()
    plt.imshow(img)
plt.show()