#dataset用于准备数据，dataloard用于以特定形式加载数据到网络等当中
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

#准备测试数据集：train=False选择了训练集，transform选了totensor是转换成tensor格式，download是是否下载数据集(true如果有数据集了就不再下载)
test_data = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=False,transform=transforms.ToTensor(),download=True)
#dataset是选择的数据集，shuffle是是否每个epoch打乱图片顺序，drop_last是batch无法整除的部分是否留下单独作为一组batch
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

#测试数据集的第一张图片数据
img, target = test_data[0]  #以图片格式-target/label的格式存储的
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):    #与上述的shuffle参数对应，每一个epoch对应一次图片乱序，使得每个回合的同一个step下的batch会不同
    step = 0              #因为for的遍历不是i，所以需要在外面设一个i以遍历不同step的图片
    for data in test_loader:    #因为需要把所有数据都打包成batch，所以用data遍历
        imgs, targets = data                                      #先赋值，再下一行的写入日志
        writer.add_images("Epoch: {}".format(epoch),imgs,step)    #因为是以batch形式加载的，所以add_images
        step = step+1

writer.close()
