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
from .cifar10_data_engine import *

class DataEngine(object):
    def __init__(
            self, 
            subsample_size, 
            root, 
            save_root, 
            dataset_name, 
            num_classes, 
            one_hot, 
            single_target_a, 
            single_target_b, 
            poison_prop, 
            num_parties, 
            num_corrupted_parties, 
            all_to_all = False,
            shuffle = True):
        self.subsample_size = subsample_size
        self.root = root
        self.save_root = save_root
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.single_target_a = single_target_a
        self.single_target_b = single_target_b
        self.poison_prop = poison_prop
        self.all_to_all = all_to_all
        self.shuffle = shuffle
        self.num_parties = num_parties
        self.num_corrupted_parties = num_corrupted_parties

    def run(self):
        train, val = self.get_dataset()
        train_data_split, train_targets_split  = format_targets_and_split(
                train, 
                self.num_classes, 
                self.one_hot, 
                self.num_parties, 
                self.shuffle)

        val_data_split, val_targets_split = format_targets_and_split(
                val, 
                self.num_classes, 
                self.one_hot, 
                self.num_parties, 
                self.shuffle)

        poison(train_data_split, 
                train_targets_split,
                self.num_corrupted_parties, 
                self.all_to_all, 
                self.one_hot, 
                self.poison_prop,
                self.single_target_a,
                self.single_target_b)

        write_data(
                train_data_split, 
                train_targets_split, 
                val_data_split,
                val_targets_split,
                self.num_parties,
                self.num_corrupted_parties,
                self.poison_prop,
                self.save_root, 
                self.dataset_name,
                self.subsample_size)

