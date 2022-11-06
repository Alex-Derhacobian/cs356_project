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


PRETRAINED_MODEL_WEIGHTS = {
        "resnet18": torchvision.models.resnet18(pretrained = True),
        "resnet34": torchvision.models.resnet34(pretrained = True),
        "resnet50": torchvision.models.resnet50(pretrained = True),
        }

MODEL_WEIGHTS = {
        "resnet18": torchvision.models.resnet18(pretrained = False),
        "resnet34": torchvision.models.resnet34(pretrained = False),
        "resnet50": torchvision.models.resnet50(pretrained = False),
        }
