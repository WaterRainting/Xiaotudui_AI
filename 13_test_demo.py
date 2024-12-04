# 用于模型泛化能力的测试，看是否判断准确
import torch
from PIL import Image
from torchvision import transforms
from fully_train_model import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "test_dog.png"
image = Image.open("test_frog.png")
image = image.convert("RGB")    # png格式的图片是4通道，这里需要转换成3通道

transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
image = transform(image)
print(image.shape)

model = torch.load("tudui_29.pth",map_location="cpu")   # note1：或者是将导入的模型换成cpu形式
print(model)

# image = image.to(device)    # note2：由于导入的模型是用cuda计算的，因此也要将input换成cuda形式（同时也要加上第7行的代码）
image = torch.reshape(image,(1,3,32,32))    # 因为原有模型的输入是4通道的，所以这里也要修改通道数，令batchsize=1

# 开启测试步骤(应该写，万一有 batchnorm 或 dropout)，同时关闭梯度，节省内存空间
model.eval()
with torch.no_grad():
    output = model(image)

print(output)
print(output.argmax(1))     #打印预测结果