from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -> tensor数据类型
# 通过transforms.ToTensor去看两个问题

# 2. 为什么我们需要Tensor数据类型：神经网络需要tensor

# 绝对路径 /Users/shaygong/Desktop/CS/ML/TuDuiPyTorch/learn_PyTorch/dataset/train/ants_image/0013035.jpg
# 相对路径 dataset/train/ants_image/0013035.jpg
img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1. transforms该如何使用(python)
tensor_trans = transforms.ToTensor() # ToTensor是一个类
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
