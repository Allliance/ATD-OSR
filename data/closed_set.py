import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import numpy as np
import random

def get_in_training_loaders(in_dataset, batch_size):

    if in_dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

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
    dataset.targets = [x for x in dataset.targets]
    
def select_indices(dataset, in_classes):
    indices = np.asarray([i for i, x in enumerate(dataset.targets) if x in in_classes])
    dataset.data = np.asarray(dataset.data)[indices]
    dataset.targets = np.asarray(dataset.targets)[indices]
    list_data(dataset)
    dataset.targets = [in_classes.index(x) for x in dataset.targets]
    

def get_in_training_loaders_osr(in_dataset, batch_size, in_classes_indices):


    if in_dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

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

    elif in_dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())

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

    elif in_dataset == 'TI':
        testset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/val', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    select_indices(testset, in_classes_indices)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader