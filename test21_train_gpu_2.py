import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from test19_model import *

# 定义训练的设备
device = torch.device("mps")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为10
print("Train data size: {}".format(train_data_size))
print("Test data size: {}".format(test_data_size))

# 利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class GST(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
gst = GST()
gst = gst.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_gst = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(gst.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-------- 第 {} 轮训练开始 --------".format(i+1))

    # 训练步骤开始
    gst.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = gst(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    gst.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataLoader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = gst(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = ((outputs.argmax(1) == targets).sum())
            total_accuracy += accuracy
    print("Total Test loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(gst, "gst_{}.pth".format(i))

writer.close()
