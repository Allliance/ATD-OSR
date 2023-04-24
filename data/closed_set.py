import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import numpy as np
import random

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


def get_in_training_loaders(in_dataset, batch_size):

    if in_dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())
    elif in_dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())
    elif in_dataset == 'fmnist':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())
    elif in_dataset == 'svhn':
        dataset = torchvision.datasets.SVHN(root='./data', split='train',
                                          download=True, transform=transforms.ToTensor())                        
    elif in_dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())
        dataset.targets = sparse2coarse(dataset.targets)

    elif in_dataset == 'TI':
        dataset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, valloader


def list_data(dataset):
    dataset.data = [x for x in dataset.data]
    try:
        dataset.targets = [x for x in dataset.targets]
    except Exception as e:
        dataset.labels = [x for x in dataset.labels]
    
def select_indices(dataset, in_classes):
    try:
        indices = np.asarray([i for i, x in enumerate(dataset.targets) if x in in_classes])
        dataset.targets = np.asarray(dataset.targets)[indices]
        dataset.data = np.asarray(dataset.data)[indices]
        list_data(dataset)
        dataset.targets = [in_classes.index(x) for x in dataset.targets]
    except Exception as e:
        indices = np.asarray([i for i, x in enumerate(dataset.labels) if x in in_classes])
        dataset.labels = np.asarray(dataset.labels)[indices]
        dataset.data = np.asarray(dataset.data)[indices]
        list_data(dataset)
        dataset.labels = [in_classes.index(x) for x in dataset.labels]
    

def get_in_training_loaders_osr(in_dataset, batch_size, in_classes_indices):


    if in_dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())
    elif in_dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())
    elif in_dataset == 'fmnist':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())
    elif in_dataset == 'svhn':
        dataset = torchvision.datasets.SVHN(root='./data', split='train',
                                          download=True, transform=transforms.ToTensor())                        
    elif in_dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())
        dataset.targets = sparse2coarse(dataset.targets)

    elif in_dataset == 'TI':
        dataset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    select_indices(dataset, in_classes_indices)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, test_size])

    print("Size of training set:", len(trainset))
    
    print("Size of validation set:", len(valset))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, valloader


def get_in_testing_loader(in_dataset, batch_size):

    if in_dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'mnist':
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())
    elif in_dataset == 'fmnist':
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())
    elif in_dataset == 'svhn':
        testset = torchvision.datasets.SVHN(root='./data', split='test',
                                          download=True, transform=transforms.ToTensor())                        
    elif in_dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())
        testset.targets = sparse2coarse(testset.targets)

    elif in_dataset == 'TI':
        testset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/val', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader

def get_in_testing_loader_osr(in_dataset, batch_size, in_classes_indices):

    if in_dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())
    elif in_dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())
        testset.targets = sparse2coarse(testset.targets)
    elif in_dataset == 'mnist':
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transforms.Compose[transforms.Grayscale(3), transforms.ToTensor()])
    elif in_dataset == 'fmnist':
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                          download=True, transform=transforms.Compose[transforms.Grayscale(3), transforms.ToTensor()])
    elif in_dataset == 'svhn':
        testset = torchvision.datasets.SVHN(root='./data', split='test',
                                          download=True, transform=transforms.ToTensor())                        
    elif in_dataset == 'TI':
        testset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/val', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    select_indices(testset, in_classes_indices)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader
