import torch
import numpy as np
import torchvision
import crypten
import sklearn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import tqdm
import copy
import argparse

DUMMY_INPUTS = {
        'MNIST': lambda batch_size : torch.empty(batch_size, 1, 28, 28)
        }

MODEL_WEIGHTS = {
        "resnet18": torchvision.models.resnet18(pretrained = False),
        "resnet34": torchvision.models.resnet34(pretrained = False),
        "resnet50": torchvision.models.resnet50(pretrained = False),
        "resnet18_pretrained": torchvision.models.resnet18(pretrained = True),
        "resnet34_pretrained": torchvision.models.resnet34(pretrained = True),
        "resnet50_pretrained": torchvision.models.resnet50(pretrained = True),
        }
