from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

# ToTensor
trans_totensor = transforms.ToTensor() # ToTensor是一个类
img_tensor = trans_totensor(img)
writer.add_image("Tensor_img", img_tensor)

# Normalize
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((12, 10))
img_resize = trans_resize(img) # resize需要PIL
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)

# Compose
trans_resize_2 = transforms.Resize((12, 10))
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(512) # 给一个数裁剪正方形，还可以trans_random = transforms.RandomCrop((256, 512))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()