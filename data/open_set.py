import cv2
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from glob import glob
import os
import random

def food_loader(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(32,32))
    return img

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root='./out', size=32, transform=transforms.Compose([transforms.ToTensor()])):
        file_paths = glob(os.path.join(root, '*.jpeg'))
        
        self.transform = transform
        self.data = [np.asarray(Image.open(x).convert('RGB').resize((size, size))) for x in file_paths]
        self.targets = np.zeros(len(self.data))

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)
        
        return image, int(target)

    def __len__(self):
        return len(self.data)

def get_out_training_loaders_osr(batch_size, size=5000, exposure_path='./out'):

    trainset_out = ImageNet(root=exposure_path, transform=transforms.Compose([transforms.ToPILImage(),
                                                                                    transforms.RandomChoice(
                                                                                        [transforms.RandomApply([transforms.RandomAffine(90, translate=(0.15, 0.15), scale=(0.85, 1), shear=None)], p=0.6),
                                                                                         transforms.RandomApply([transforms.RandomAffine(0, translate=None, scale=(0.5, 0.75), shear=30)], p=0.6),
                                                                                         transforms.RandomApply([transforms.AutoAugment()], p=0.9),]),
                                                                                    transforms.ToTensor()]))
    trainset_out.data = random.sample(trainset_out.data, k=size)
    
    trainloader_out = DataLoader(trainset_out, batch_size=batch_size, shuffle=True, num_workers=2)

    valset_out = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
    valloader_out = DataLoader(valset_out, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader_out, valloader_out

def select_indices(dataset, in_classes):
    try:
        indices = np.asarray([i for i, x in enumerate(dataset.targets) if x in in_classes])
        if type(dataset.data) == list:
            dataset.data = np.asarray(dataset.data)[indices]
            dataset.data = [x for x in dataset.data]
        else:
            dataset.data = dataset.data[indices]
            
        if type(dataset.targets) == list:
            dataset.targets = np.asarray(dataset.targets)[indices]
            dataset.targets = [x for x in dataset.targets]
        else:
            dataset.targets = dataset.targets[indices]
    except Exception as e:
        indices = np.asarray([i for i, x in enumerate(dataset.labels) if x in in_classes])
        if type(dataset.data) == list:
            dataset.data = np.asarray(dataset.data)[indices]
            dataset.data = [x for x in dataset.data]
        else:
            dataset.data = dataset.data[indices]
            
        if type(dataset.labels) == list:
            dataset.labels = np.asarray(dataset.labels)[indices]
            dataset.labels = [x for x in dataset.labels]
        else:
            dataset.labels = dataset.labels[indices]
    

def get_out_training_loaders(batch_size):

    trainset_out = torchvision.datasets.ImageFolder(root = 'data/food-101/images/', loader=food_loader, 
                                                    transform = transforms.Compose([transforms.ToPILImage(),
                                                                                    transforms.RandomChoice(
                                                                                        [transforms.RandomApply([transforms.RandomAffine(90, translate=(0.15, 0.15), scale=(0.85, 1), shear=None)], p=0.6),
                                                                                         transforms.RandomApply([transforms.RandomAffine(0, translate=None, scale=(0.5, 0.75), shear=30)], p=0.6),
                                                                                         transforms.RandomApply([transforms.AutoAugment()], p=0.9),]),
                                                                                    transforms.ToTensor(),  ]))
    trainloader_out = DataLoader(trainset_out, batch_size=batch_size, shuffle=True, num_workers=2)

    valset_out = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
    valloader_out = DataLoader(valset_out, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader_out, valloader_out


def bird_loader(path):
    path = path.split('/')
    if path[-1][0:2] == '._':
        path[-1] = path[-1][2:]
    path = '/'.join(path)
    img=cv2.imread(path)
    img=cv2.resize(img,(32,32))
    return img

def flower_loader(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(32,32))
    return img


def get_out_testing_datasets_osr(in_dataset, out_indices):

    out_datasets = []
    returned_out_names = []


    if in_dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'mnist':
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transforms.Compose([transforms.Resize(32), transforms.Grayscale(3), transforms.ToTensor()]))
    elif in_dataset == 'fmnist':
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                          download=True, transform=transforms.Compose([transforms.Resize(32), transforms.Grayscale(3), transforms.ToTensor()]))
    elif in_dataset == 'svhn':
        testset = torchvision.datasets.SVHN(root='./data', split='test',
                                          download=True, transform=transforms.ToTensor())                        
    elif in_dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())
        testset.targets = sparse2coarse(testset.targets)
        
    select_indices(testset, out_indices)


    out_datasets.append(testset)
    returned_out_names.append(f"{in_dataset} out classes: " + str(out_indices))
    
    return returned_out_names, out_datasets


def get_out_testing_datasets(out_names):

    out_datasets = []
    returned_out_names = []

    for name in out_names:

        if name == 'mnist':
            mnist = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                  transforms.Resize(32),
                                                                                                  transforms.Lambda(lambda x : x.repeat(3, 1, 1)),
                                                                                                  ]))
            returned_out_names.append(name)
            out_datasets.append(mnist)
        
        elif name == 'tiny_imagenet':
            tiny_imagenet = torchvision.datasets.ImageFolder(root = 'data/tiny-imagenet-200/test', transform=transforms.Compose([transforms.ToTensor(),
                                                                                                          transforms.Resize(32)]))
            
            returned_out_names.append(name)
            out_datasets.append(tiny_imagenet)
        
        elif name == 'places':
            places365 = torchvision.datasets.Places365(root = 'data/', split = 'val', small = True, download = False, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                          transforms.Resize(32)]))

            returned_out_names.append(name)
            out_datasets.append(places365)
        
        elif name == 'LSUN':
            LSUN = torchvision.datasets.ImageFolder(root = 'data/LSUN_resize/', transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(LSUN)

        elif name == 'iSUN':
            iSUN = torchvision.datasets.ImageFolder(root = 'data/iSUN/', transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(iSUN)
          
        elif name == 'birds': 
            birds = torchvision.datasets.ImageFolder(root = 'data/images/', loader=bird_loader, transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(birds)
        
        elif name == 'flowers':
            flowers = torchvision.datasets.ImageFolder(root = 'data/flowers/', loader=flower_loader, transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(flowers)
          
        elif name == 'coil':
            coil_100 = torchvision.datasets.ImageFolder(root = 'data/coil/', transform=transforms.Compose([transforms.ToTensor(),
                                                                                                transforms.Resize(32)]))
            
            returned_out_names.append(name)
            out_datasets.append(coil_100)
        
        else:
          print(name, ' dataset is not implemented.')
    
    return returned_out_names, out_datasets
