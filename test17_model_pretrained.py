import torchvision.datasets
from torch import nn

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
vgg16_false = torchvision.models.vgg16(weights=None)   # 老版：pretrained=False
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')

print(vgg16_true)

dataset = torchvision.datasets.CIFAR10(root='../dataset', download=False, train=True,
                                       transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)
vgg16_false.classifier[6] = nn.Linear(4096, 10)