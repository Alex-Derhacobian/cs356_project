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

class MNISTDataEngine(DataEngine):
    def __init__(self, *args, **kwargs):
        super(MNISTDataEngine, self).__init__(*args, **kwargs)
    

    def get_dataset(self):

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        mnist_train = torchvision.datasets.MNIST(root = self.root,
                                                download = True,
                                                train = True,
                                                transform = transform)
        mnist_val = torchvision.datasets.MNIST(root = self.root, 
                                            download = True,
                                            train = False,
                                            transform = transform)

        subsample_train_ids = np.random.randint(0, 
                len(mnist_train), 
                int(len(mnist_train)*self.subsample_size))

        subsample_val_ids = np.random.randint(0, 
                len(mnist_val), 
                int(len(mnist_val)*self.subsample_size))

        mnist_train = torch.utils.data.Subset(mnist_train, subsample_train_ids)
        mnist_val = torch.utils.data.Subset(mnist_val, subsample_val_ids)

        return mnist_train, mnist_val

                
