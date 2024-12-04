#Transforms主要用于图像的处理，比如处理成一维数据或者归一化等
#使用方法：
# 1、先实例化对应方法的类
# 2、根据官方文档，修改参数的格式，使得其对应了函数的输入要求，比如是PEL还是tensor还是narray
# 3、给实例的__call__函数传入参数
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")
image = Image.open("fdu6.jpg")
print(image)

#Totensor：将PIL Image or numpy.ndarray转换成tensor的格式
trans_totensor = transforms.ToTensor()  #创建一个ToTensor类，以便后续操作
img_tensor = trans_totensor(image)      #trans_totensor()写了__call__，所以可以直接放参数
                                        #__call__可以让函数trans_totensor(xxxxxx)的形式运行，而不需要trans_totensor.__call__(image)
writer.add_image("ToTensor",img_tensor)

#Normalize：归一化数据
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  #三通道，[通道1均值,通道2均值,通道3均值],[通道1标准差,通道2标准差,通道3标准差]
img_norm = trans_norm(img_tensor)   #输入类型需要是tensor类型
writer.add_image("Normalize",img_norm)
print(img_norm[0][0][0])

#Resize：改变图片的形状，如长和宽等
print(image.size)
trans_resize = transforms.Resize([256,1048]) #[长，宽]或者[原始图像的最小边变成的值（即简单的缩放）]
img_resize = trans_resize(image)     #输入类型需要是PIL或者tensor类型,此处是PIL
img_resize_tensor = trans_totensor(img_resize)
writer.add_image("Resize",img_resize_tensor,3)
print(img_resize.size)

#Compose：将多种transforms操作和为一体，要注意上一个操作的输出和下一个操作的输入所需类型要对齐
trans_resize_2 = transforms.Resize(256)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor]) #先resize再totensor，用一行函数完成
img_resize_2 = trans_compose(image)   #注意这里的逻辑，因为是compose代替了两个函数，所以输入是最初的image了
writer.add_image("Compose",img_resize_2)

#RandCrop：随机裁剪，从原有图像中随机切出特定大小的图像块
trans_randomcrop = transforms.RandomCrop(512)
for i in range(10): #每次循环都要重新切，然后把切的写到tensorboard里面
    img_randomcrop = trans_randomcrop(img_tensor)  # 需要的输入是tensor
    writer.add_image("RandomCrop", img_randomcrop,i)

writer.close()


