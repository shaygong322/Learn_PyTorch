import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root='../dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                        download=False)  # 已下载好的重新访问
dataloader = DataLoader(test_set, batch_size=64)

class GST(nn.Module):
    def __init__(self):
        super(GST, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

gst = GST()

writer = SummaryWriter('../logs')
step = 0
for data in dataloader:
    imgs, targets = data
    output = gst(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30]) 6个channel无法显示, 只能3个
    output = torch.reshape(output, ([-1, 3, 30, 30]))
    writer.add_images("output", output, step)
    step += 1

writer.close()
