import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

trainset = torchvision.datasets.CIFAR10(root='/mnt/liguanlin/DataSets/cifar', train=True,
                                        download=True,transform=None)

testset = torchvision.datasets.CIFAR10(root='/mnt/liguanlin/DataSets/cifar', train=False,
                                       download=True, transform=None)

train_set_size = len(trainset)
print(train_set_size)

test_set_size = len(testset)
print(test_set_size)

print(type(trainset[0])) # <class 'tuple'>

trainset_labels = np.zeros(train_set_size)
print(trainset_labels.shape)

testset_labels = np.zeros(test_set_size)
print(testset_labels.shape)

for i in range(train_set_size):
    img_path = '/mnt/liguanlin/DataSets/cifar/train/{i}.png'.format(i=i+1)
    sample = trainset[i][0]
    sample.save(img_path)
    trainset_labels[i] = trainset[i][1]

for i in range(test_set_size):
    img_path = '/mnt/liguanlin/DataSets/cifar/test/{i}.png'.format(i=i+1)
    img = testset[i][0]
    img.save(img_path)
    testset_labels[i] = testset[i][1]

np.savetxt(fname='/mnt/liguanlin/DataSets/cifar/train/trainset_label.csv', X= trainset_labels, delimiter=',')
np.savetxt(fname='/mnt/liguanlin/DataSets/cifar/test/testset_label.csv', X= testset_labels, delimiter=',')

trainset_labels_read = np.loadtxt("/mnt/liguanlin/DataSets/cifar/train/trainset_label.csv", delimiter=',')
testset_labels_read = np.loadtxt("/mnt/liguanlin/DataSets/cifar/test/testset_label.csv", delimiter=',')
print(trainset_labels_read.shape)
print(testset_labels_read.shape)

print(trainset_labels_read[0])
print(trainset_labels_read[1])
print(testset_labels_read)

#https://cloud.tencent.com/developer/article/1144751
