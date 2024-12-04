#Tensorboard主要用于Loss下降过程和神经网络训练过程图片的可视化
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")  #这里writer就是一个写日志的SummaryWriter类，用于实现TensorBoard
img_path = "dataset/train/ants_image/7759525_1363d24e88.jpg"
img_PIL = Image.open(img_path)  #先获取PIL格式的图片
img_numpy = np.array(img_PIL)   #由于add_image()的第二项需要np的格式，所以需要进行转换
print(type(img_numpy))
print(img_numpy.shape)

#图片的显示
#这里dataformats需要修改，因为上面print的shape是高-宽-通道的格式，而默认的是通道-高-宽的形式
#所以需要重写这个参数，告诉函数dataforms是HWC的格式
writer.add_image("ant_image_train",img_numpy,1,dataformats="HWC")

#函数图像、Loss的显示
for i in range(100):
    writer.add_scalar("y=4x",4*i,i)

writer.close()
