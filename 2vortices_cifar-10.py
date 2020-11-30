# 重新使用cifar-10训练数据训练一个网络
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from keras import optimizers as kopt
from keras.models import Sequential
from keras.layers import Dense
import os
from skimage import data_dir, io, transform, color

# Geometry definition
N=28
x = np.linspace(-5, 5-10/N, N)
[X,Y] = np.meshgrid(x,x)
thet = np.arctan2(Y,X)

# 读取D:/DataSets/PennFudanPed/28_28/文件夹下的数据，并使用已训练好的网络进行处理
# 读取图片作为输入训练网络
cifar_image_root_dir = 'D:/DataSets/cifar10/28_28/train'
# image_path = os.path.join(image_root_dir, self.type_name)
cifar_img_name_list = os.listdir(cifar_image_root_dir)  # 得到path目录下所有图片名称的一个list

cifar_image_list = []

image_count = len(cifar_img_name_list)
print('image_count:')
print(image_count)
for i in range(image_count):
    img_item_path = os.path.join(cifar_image_root_dir, cifar_img_name_list[i])
    img = io.imread(img_item_path)
    cifar_image_list.append(img)

# 将list转换为ndarray
cifar_image_array_train = np.array(cifar_image_list)
print(cifar_image_array_train.shape)
print(type(cifar_image_array_train))

# Normalize datasets
cifar_image_array_train = cifar_image_array_train.astype('float32') / 255

# Allocate the memory for vortex transformed intensity patterns
cifar_A = np.zeros([cifar_image_array_train.shape[0], 28, 28])
cifar_B = np.zeros([cifar_image_array_train.shape[0], 28, 28])
cifar_C = np.zeros([cifar_image_array_train.shape[0], 28, 28])

# Wide Gaussian, you can change this parameter and see what happens
rad = 22
G = np.exp(-(X ** 2 + Y ** 2) / rad)
# Top charges
m1 = 1
m2 = 3
m3 = 5
# Vortices themselves
V1 = np.exp(1j * m1 * thet)
V2 = np.exp(1j * m2 * thet)
V3 = np.exp(1j * m3 * thet)
# Gaussian modulated vortices
VG1 = V1
VG2 = V2
VG3 = V3
# Iamginary Pi
impi = 1j * np.pi
# The cycle that computes the intensity patterns
# Computationally expensive, I would recommend saving the A, B and C after they are computed for the first time
# The 100000 factor is used to normalize the inputs to CNN
# Create the training data
scale = 1000
for k in range(0, cifar_image_array_train.shape[0]):
    TF1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG1 * np.exp(impi * cifar_image_array_train[k]))))
    TF2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG2 * np.exp(impi * cifar_image_array_train[k]))))
    TF3 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG3 * np.exp(impi * cifar_image_array_train[k]))))
    cifar_A[k] = ((np.abs(TF1)) ** 2) / scale
    cifar_B[k] = ((np.abs(TF2)) ** 2) / scale
    cifar_C[k] = ((np.abs(TF3)) ** 2) / scale

# Flatten the training set
cifar_xr = cifar_image_array_train.reshape(cifar_image_array_train.shape[0], 28 ** 2)

# Flatten the arrays and stack two of them together

cifar_A = cifar_A.reshape(cifar_image_array_train.shape[0], 28 ** 2)
cifar_B = cifar_B.reshape(cifar_image_array_train.shape[0], 28 ** 2)

cifar_Cd = np.hstack((cifar_A, cifar_B))
print(cifar_Cd.shape)

# Create the model with two layers

# The performance depends on the version of Tensorflow
# The smallest MSE does not correspond to the smallest SSIM (True for the linear model). You might want to stop training earlier
ep = 5
# Linear-Non-linear model is the simplest, but might require additional training
# Fully-linear needs threholding and tinkering with number of epochs, but achieves higher SSIM
# The greedy models with 4 or 5/6 vortices need more hidden units, more training, maybe dropout layer. A little bit of a
# headache.
# Number of nodes in hidden layer
Nh = 1400
cifar_model = Sequential()
cifar_model.add(Dense(Nh, input_dim=2 * 28 ** 2, activation='linear', use_bias=False))
# model.add(Dropout(0.06))
cifar_model.add(Dense(28 ** 2, input_dim=Nh, activation='relu', use_bias=True))
# For fully linear model we use threshold to get rid of "background"
cifar_model.compile(loss='mean_squared_error', optimizer='adam')
# Training section
St_tr = 30
En_tr = 50000
cifar_story = cifar_model.fit(cifar_Cd[St_tr:En_tr] / 100, cifar_xr[St_tr:En_tr], epochs=ep)

# 使用已训练好的网络测试cafir
import os

# 读取D:/DataSets/PennFudanPed/28_28/文件夹下的数据，并使用已训练好的网络进行处理
# 读取图片作为输入训练网络
image_root_dir = 'D:/DataSets/cifar10/28_28/test'
# image_path = os.path.join(image_root_dir, self.type_name)
img_name_list = os.listdir(image_root_dir)  # 得到path目录下所有图片名称的一个list

image_list = []

image_count = len(img_name_list)
print('image_count:')
print(image_count)
for i in range(image_count):
    img_item_path = os.path.join(image_root_dir, img_name_list[i])
    img = io.imread(img_item_path)
    image_list.append(img)

# 将list转换为ndarray
image_array_test = np.array(image_list)
print(image_array_test.shape)
print(type(image_list))

image_array_test = image_array_test.astype('float32') / 255

# Allocate the memory for vortex transformed intensity patterns (test data)
At_PennFudanPed = np.zeros([image_array_test.shape[0], 28, 28])
Bt_PennFudanPed = np.zeros([image_array_test.shape[0], 28, 28])
Ct_PennFudanPed = np.zeros([image_array_test.shape[0], 28, 28])

# Create the test data
scale = 1000
for k in range(0, image_array_test.shape[0]):
    TF1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG1 * np.exp(impi * image_array_test[k]))))
    TF2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG2 * np.exp(impi * image_array_test[k]))))
    TF3 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG3 * np.exp(impi * image_array_test[k]))))
    At_PennFudanPed[k] = ((np.abs(TF1)) ** 2) / scale
    Bt_PennFudanPed[k] = ((np.abs(TF2)) ** 2) / scale
    Ct_PennFudanPed[k] = ((np.abs(TF3)) ** 2) / scale

# Flatten the arrays and stack two of them together

At_PennFudanPed = At_PennFudanPed.reshape(image_array_test.shape[0], 28 ** 2)
Bt_PennFudanPed = Bt_PennFudanPed.reshape(image_array_test.shape[0], 28 ** 2)

Cdt_PennFudanPed = np.hstack((At_PennFudanPed, Bt_PennFudanPed))
print(Cdt_PennFudanPed.shape)

# Use model to predict the shapes
St_te = 0
En_te = image_count
x_pr_PennFudanPed = cifar_model.predict(Cdt_PennFudanPed[St_te:En_te] / 100)
x_pred_PennFudanPed = x_pr_PennFudanPed.reshape(x_pr_PennFudanPed.shape[0], 28, 28)

plt.figure(figsize=(10, 10))
# vlist= [4, 8, 9, 3, 16, 22, 15, 14, 19]
z = 1
for i in range(0, 25):
    plt.subplot(5, 5, z)
    plt.imshow(x_pred_PennFudanPed[i], cmap='inferno')
    # Threshold used for linear model
    # plt.imshow(x_pred[i]*(x_pred[i]>0.2), cmap='inferno')
    plt.axis('off')
    z += 1

plt.show()

#计算ssim

from skimage.metrics import structural_similarity as ssim

dss = np.zeros((x_pred_PennFudanPed.shape[0], 1))
# Create a comparison set, from a test set
cmp_set = image_array_test[St_te:En_te]
for k in range(0, (En_te - St_te)):
    # For linear model we need a thresholding operation, for linear-non-linear it's not necessary
    # x_pred[k]*(x_pred[k]>0.18)
    dss[k] = ssim(x_pred_PennFudanPed[k], cmp_set[k])

SSIM = np.mean(dss)
print(SSIM)