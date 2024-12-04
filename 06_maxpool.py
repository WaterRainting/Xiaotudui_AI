#池化层可以减少每层的数据量，在保证模型特征保留的前提下加快运算等，这里是maxpooling
import torch
import torchvision
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import nn

dataset = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=False,transform=transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, target = data
    output = tudui(imgs)
    writer.add_images("imput", imgs, step)
    writer.add_images("maxpool",output,step)
    step = step+1

writer.close()
