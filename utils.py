#utils.py

import torch
from torchvision import transforms

from robustbench import load_model
from robustbench.model_zoo.architectures.dm_wide_resnet import DMPreActResNet, Swish
import numpy as np
import random
import torch
from data.constants import OSR_DATASETS

from models.preact_resnet import ti_preact_resnet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def list_data(dataset):
    dataset.data = [x for x in dataset.data]
    dataset.targets = [x for x in dataset.targets]
    
def select_indices(dataset, in_classes):
    indices = np.asarray([i for i, x in enumerate(dataset.targets) if x in in_classes])
    dataset.data = np.asarray(dataset.data)[indices]
    dataset.targets = np.asarray(dataset.targets)[indices]
    list_data(dataset)
    dataset.targets = [in_classes.index(x) for x in dataset.targets]

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


#normilizer model
class normalizer():
    def __init__(self, model):
        self.model = model
        self.transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    def __call__(self, x):
        out = self.model(self.transform(x))
        return out


def read_in_indices(run_index):
  indices = []
  with open(f'indices-{run_index}.osr', 'r') as f:
    lst = f.read()
    indices = [int(x) for x in lst.split(" ")]
  return indices


def get_feature_extractor_model(training_type, in_dataset, args):

    if training_type == 'adv':
    
        if in_dataset in OSR_DATASETS:
            checkpoint = torch.load(args.fea_path)
            model = DMPreActResNet(num_classes=args.num_classes, activation_fn=Swish)

            model.load_state_dict(checkpoint['model_state_dict'])

            model = model.to(device)

            # model = load_model(model_name='Rade2021Helper_R18_extra', dataset='cifar10', threat_model='Linf').to(device)
            model.logits = torch.nn.Sequential()
            model.eval()

        elif in_dataset == 'cifar100':
            model = load_model(model_name='Rade2021Helper_R18_ddpm', dataset='cifar100', threat_model='Linf').to(device)
            model.logits = torch.nn.Sequential()
            model.eval()
        
        elif in_dataset == 'TI':
            ckpt = torch.load("models/weights-best-TI.pt")['model_state_dict']
            
            model = ti_preact_resnet('preact-resnet18', num_classes=200).to(device)    
            model = torch.nn.Sequential(model)
            model = torch.nn.DataParallel(model).to(device)
            
            model.load_state_dict(ckpt)
            model.module[0].linear = torch.nn.Sequential()
            model.eval()
        
    elif training_type == 'clean':

        if in_dataset == 'cifar10':
            model_temp = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True).to(device)

        elif in_dataset == 'cifar100':
            model_temp = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True).to(device)
        
        model_temp.classifier = torch.nn.Sequential()
        model_temp.eval()
        
        model = normalizer(model_temp)
    
    return model