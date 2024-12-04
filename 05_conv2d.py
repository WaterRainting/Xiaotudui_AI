#Conv2d主要用于二维图像的卷积处理
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=False,transform=transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        #定义卷积函数参数(这里没定义kernel的参数，因此这里的kernel是随机生成的)
        self.cov1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):  #用于前向传播，调用各种卷积函数的过程
        x = self.cov1(x)
        return x

tudui = Tudui()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, target = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    # 因为会输出（64，6，30，30），而add_images只能是3通道数据，因此reshape成3通道数据
    output_reshape = torch.reshape(output,(-1,3,30,30))
    writer.add_images("input",imgs,step)
    writer.add_images("outputs",output_reshape,step)
    step = step+1

writer.close()
