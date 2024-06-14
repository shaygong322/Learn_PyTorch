import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class GST(nn.Module):
    def __init__(self):
        super(GST, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

gst = GST()
writer = SummaryWriter('../logs_maxpool')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step) # !!!注意是images, 有个s
    output = gst(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()