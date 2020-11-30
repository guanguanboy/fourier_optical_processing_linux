import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torch import nn, optim
import time
import os
from models import dnn

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load datasets - clothes and numbers
# Since only one is used in training you can comment one of them
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

train_set = torchvision.datasets.FashionMNIST(root='/mnt/liguanlin/DataSets/Datasets/FashionMNIST',
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]))

#如果要获取到fashionMNIST的测试集则
test_set = torchvision.datasets.FashionMNIST(root='/mnt/liguanlin/DataSets/Datasets/FashionMNIST',
                                              train=False,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]))

#print(train_set.data)
#print((train_set.data).shape)
#print(train_set.targets)

x_train = (train_set.data).numpy()
y_train = (train_set.targets).numpy()
x_test = (test_set.data).numpy()
y_test = (test_set.targets).numpy()

"""
print(type(x_train))
print(type(y_train))
print(x_train.shape)
print(y_train.shape)


结果如下：
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
(60000, 28, 28)
(60000,)
"""


"""
print(type(x_test))
print(type(y_test))
print(x_test.shape)
print(y_test.shape)
结果如下：
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
(10000, 28, 28)
(10000,)
"""

import matplotlib.pyplot as plt

# Geometry definition
N=28
x = np.linspace(-5, 5-10/N, N)
[X,Y] = np.meshgrid(x,x)
thet = np.arctan2(Y,X)

# Normalize datasets
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# Allocate the memory for vortex transformed intensity patterns
A = np.zeros([x_train.shape[0],28,28])
B = np.zeros([x_train.shape[0],28,28])
C = np.zeros([x_train.shape[0],28,28])

# Wide Gaussian, you can change this parameter and see what happens
rad =22
G = np.exp( -(X**2 + Y**2)/rad)
# Top charges
m1=1
m2=3
m3=5
#Vortices themselves
V1=np.exp(1j*m1*thet)
V2=np.exp(1j*m2*thet)
V3=np.exp(1j*m3*thet)
# Gaussian modulated vortices
VG1 = V1
VG2 = V2
VG3 = V3
# Iamginary Pi
impi = 1j*np.pi # 1j 在python中表示一个虚部为1的虚数

# The cycle that computes the intensity patterns
# Computationally expensive, I would recommend saving the A, B and C after they are computed for the first time
# The 100000 factor is used to normalize the inputs to CNN
# Create the training data
scale = 1000
for k in range(0,x_train.shape[0]):
    TF1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG1*np.exp(impi*x_train[k]))))
    TF2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG2*np.exp(impi*x_train[k]))))
    TF3 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG3*np.exp(impi*x_train[k]))))
    A[k]=((np.abs(TF1))**2)/scale
    B[k]=((np.abs(TF2))**2)/scale
    C[k]=((np.abs(TF3))**2)/scale



# Allocate the memory for vortex transformed intensity patterns (test data)
At = np.zeros([x_test.shape[0],28,28])
Bt = np.zeros([x_test.shape[0],28,28])
Ct = np.zeros([x_test.shape[0],28,28])

# Create the test data
scale = 1000
for k in range(0,x_test.shape[0]):
    TF1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG1*np.exp(impi*x_test[k]))))
    TF2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG2*np.exp(impi*x_test[k]))))
    TF3 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG3*np.exp(impi*x_test[k]))))
    At[k]=((np.abs(TF1))**2)/scale
    Bt[k]=((np.abs(TF2))**2)/scale
    Ct[k]=((np.abs(TF3))**2)/scale


# Flatten the training set
xr = x_train.reshape(x_train.shape[0], 28**2)

# Flatten the arrays and stack two of them together

A = A.reshape(x_train.shape[0], 28 ** 2)
B = B.reshape(x_train.shape[0], 28 ** 2)

Cd = np.hstack((A, B))

#print('Cd.shape:')
#print(Cd.shape)

Cd_show = Cd.reshape(x_train.shape[0], 28, 56)
#print('Cd_show.shape:')
#print(Cd_show.shape)
#print(type(Cd_show))
# Flatten the arrays and stack two of them together


At = At.reshape(x_test.shape[0], 28 ** 2)
Bt = Bt.reshape(x_test.shape[0], 28 ** 2)

Cdt = np.hstack((At, Bt))
Cdt_show = Cdt.reshape(x_test.shape[0], 28, 56)
#print(Cdt_show[0])

"""
由于转换之后，Cdt_show是浮点类型的，所以如果要将其保存成png格式的话
会出现如下的错误：
OSError: cannot write mode F as PNG

im = Image.fromarray(Cdt_show[0], mode='F')
im.save("./datasets_2vortex/first_test.png")
"""

"""
print('Cdt_show.shape: ')
print(Cdt_show.shape)

print('Cdt.shape: ')
print(Cdt.shape)

print('Cdt.type: ')
print(type(Cdt))

print('x_test_type:')
print(type(x_test))

print(x_test.shape)
"""

plt.figure('original', figsize = (10,10))
#vlist= [4, 8, 9, 3, 16, 22, 15, 14, 19]
z=1
for i in range(0,25):
    plt.subplot(5,5,z)
    plt.imshow(Cdt_show[i], cmap='inferno')
    # Threshold used for linear model
    #plt.imshow(x_pred[i]*(x_pred[i]>0.2), cmap='inferno')
    plt.axis('off')
    z+=1

plt.show()

#直接构造Datasets 和 Dataloader

#可以直接利用上面解析好的全局变量。
#如果不能使用全局变量，则可以先把mnist数据集中的数据保存成图片，然后在getitem函数中将其转换为vortex模式。
class MyVortexMnistDataSet(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data_aray = data
        self.label_array = labels
        self.transform = transform

    def __getitem__(self, item):
        image = self.data_aray[item]
        label = self.label_array[item]
        #print('image type:')
        #print(type(image))
        image = image.astype(np.float32)
        if self.transform is not None: #如果transform不等于None,那么执行转换
            image = self.transform(image)
            #label = self.transform(label)

        return image, label

    def __len__(self):
        return len(self.label_array)

# 展平图像
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class GlobalAvgPool2d(nn.Module):
    """
    全局平均池化层
    可通过将普通的平均池化的窗口形状设置成输入的高和宽实现
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    net.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train()  # 改回训练模式
    return acc_sum / n


def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, model_path):
    global best_test_acc
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            #print('y_hat and y type:')
            #print(type(y_hat))
            #print(type(y))
            #print(y_hat.shape)
            #print(y.shape)
            #print(y_hat)
            #print(y)

            l = loss(y_hat, y.long())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        if test_acc > best_test_acc:
            print('find best! save at model/best.pth')
            best_test_acc = test_acc
            torch.save(net.state_dict(), '{}/model/dnn_best.pth'.format(model_path))


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToPILImage(),  # 不转换为PIL会报错
        transforms.ToTensor(),
        transforms.Normalize([0.40], [1.89])
    ])
    train_dataset = MyVortexMnistDataSet(Cd_show, y_train, transform)

    batch_size = 64
    num_workers = 4

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    image, labels = next(iter(train_dataloader))
    #print("aa:")
    #print(image)
    #print(labels)
    #print(image.shape)
    #print(labels.shape)
    #print(image.type)
    #print(labels.type)

    test_dataset = MyVortexMnistDataSet(Cdt_show, y_test, transform)

    #print(getStat(train_dataset, 1)) #计算得([0.39978203], [1.890278])
    #print(getStat(test_dataset, 1)) #计算得([0.3998611], [1.8868754])

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    train_iter = train_dataloader
    test_iter = test_dataloader

    num_inputs = 28*56
    num_hiddens = 4096
    num_outputs = 10

    # 训练过程
    net = dnn.DNN(num_inputs, num_hiddens, num_outputs)
    print(net)
    # model_path = "2_baseline"
    # net.load_state_dict(torch.load('{}/model/best.pth'.format(model_path)))
    net = net.to(device)

    best_test_acc = 0
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    lr = 0.001
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    lr, num_epochs = 0.001, 100
    print('训练...')
    model_path = '.'
    train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, model_path)
