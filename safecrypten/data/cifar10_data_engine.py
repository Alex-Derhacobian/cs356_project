import torch
import numpy as np
import torchvision
import crypten
import sklearn
import matplotlib.pyplot as plt
from torchvision import transforms
import tqdm
import copy
import argparse
from .utils import *
from .data_engine import DataEngine

class CIFAR10DataEngine(DataEngine):
    def __init__(self, *args, **kwargs):
        super(CIFAR10DataEngine, self).__init__(*args, **kwargs)
    

    def get_dataset(self):

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cifar10_train = torchvision.datasets.CIFAR10(root = self.root,
                                                download = True,
                                                train = True,
                                                transform = transform)
        cifar10_val = torchvision.datasets.CIFAR10(root = self.root, 
                                            download = True,
                                            train = False,
                                            transform = transform)

        subsample_train_ids = np.random.randint(0, 
                len(cifar10_train), 
                int(len(cifar10_train)*self.subsample_size))

        subsample_val_ids = np.random.randint(0, 
                len(cifar10_val), 
                int(len(cifar10_val)*self.subsample_size))

        cifar10_train = torch.utils.data.Subset(cifar10_train, subsample_train_ids)
        cifar10_val = torch.utils.data.Subset(cifar10_val, subsample_val_ids)

        return cifar10_train, cifar10_val

                
