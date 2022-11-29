import torch
import glob
import os
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
from .utils import *
from .dataowner import *

class SafeNetEngine:
    def __init__(
            self, 
            model_arch, 
            dataset_name, 
            num_parties, 
            num_corrupted_parties, 
            poison_prop,
            data_root_dir,
            acc_threshold, 
            testset_subsample_size,
            dataowner_models_root_dir = None):

        self.model_arch = model_arch
        self.dataset_name = dataset_name
        self.num_parties = num_parties
        self.num_corrupted_parties = num_corrupted_parties
        self.poison_prop = poison_prop
        self.data_root_dir = data_root_dir
        self.acc_threshold = acc_threshold
        self.testset_subsample_size = testset_subsample_size
        self.dataowner_models_root_dir = dataowner_models_root_dir

    def get_testset(self):

        all_val_data = torch.empty((0))
        all_val_targets = torch.empty((0))

        for index in range(self.num_parties):
            val_raw_dataset_dir = os.path.join(self.data_root_dir, 
                    f"{self.num_parties}/"
                    f"{self.num_corrupted_parties}_corrupted/"
                    f"{self.poison_prop}_poisoned/val")


            val_data_pickled = os.path.join(val_raw_dataset_dir, f"data_{index}.pth")
            val_targets_pickled = os.path.join(val_raw_dataset_dir, f"targets_{index}.pth")

            val_data = torch.load(val_data_pickled)
            val_targets = torch.load(val_targets_pickled)

            all_val_data = torch.cat([all_val_data, val_data], dim = 0)
            all_val_targets = torch.cat([all_val_targets, val_targets], dim = 0)

        #Random subsampling
        target_mask = np.arange(all_val_data.shape[0])
        np.random.shuffle(target_mask)
        target_mask = target_mask[:int(all_val_data.shape[0]*self.testset_subsample_size)]

        test_data = all_val_data[target_mask]
        test_targets = all_val_data[target_mask]
        testset = torch.utils.data.TensorDataset(test_data, test_targets)
        return testset

    def joint_predict(self):
        print(f"Performing Joint Prediction with {len(self.dataowners)} DataOwners")
        testset = self.get_testset()
        all_dataowner_outputs = torch.empty((0))
        for dataowner in self.dataowners:
            dataowner_outputs = torch.unsqueeze(dataowner.predict(testset), dim = 1)
            all_dataowner_outputs = torch.cat([all_dataowner_outputs, dataowner_outputs], dim = 1)

        print(all_dataowner_outputs)
        joint_predictions = torch.mode(all_dataowner_outputs, dim = 1).values
        print(joint_predictions)

        

    def get_dataowner_dataset(self, index):
        train_raw_dataset_dir = os.path.join(self.data_root_dir, 
                f"{self.num_parties}/"
                f"{self.num_corrupted_parties}_corrupted/"
                f"{self.poison_prop}_poisoned/train")

        val_raw_dataset_dir = os.path.join(self.data_root_dir, 
                f"{self.num_parties}/"
                f"{self.num_corrupted_parties}_corrupted/"
                f"{self.poison_prop}_poisoned/val")


        assert(os.path.isdir(train_raw_dataset_dir))
        assert(os.path.isdir(val_raw_dataset_dir))

        train_data_pickled = os.path.join(train_raw_dataset_dir, f"data_{index}.pth")
        train_targets_pickled = os.path.join(train_raw_dataset_dir, f"targets_{index}.pth")

        val_data_pickled = os.path.join(val_raw_dataset_dir, f"data_{index}.pth")
        val_targets_pickled = os.path.join(val_raw_dataset_dir, f"targets_{index}.pth")

        train_data = torch.load(train_data_pickled)
        train_targets = torch.load(train_targets_pickled)

        val_data = torch.load(val_data_pickled)
        val_targets = torch.load(val_targets_pickled)

        dataowner_train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
        dataowner_val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)

        return dataowner_train_dataset, dataowner_val_dataset

    def get_dataowner_models(self):
        return glob.glob(os.path.join(self.dataowner_models_root_dir, "dataowner=*"))

    def get_global_val_dataset(self):

        global_val_data = torch.empty((0))
        global_val_targets = torch.empty((0))

        for dataowner in self.dataowners:
            dataowner_val_dataset= dataowner.val_dataset
            dataowner_val_data = dataowner_val_dataset.tensors[0]
            dataowner_val_targets = dataowner_val_dataset.tensors[1]

            global_val_data = torch.cat((global_val_data, dataowner_val_data), dim = 0)
            global_val_targets = torch.cat((global_val_targets, dataowner_val_targets), dim = 0)

        global_val_dataset = torch.utils.data.TensorDataset(global_val_data, global_val_targets)

        return global_val_dataset

    def filter_ensembles(self):
        global_val_dataset = self.get_global_val_dataset()

        valid_dataowners = []

        print("Filtering Ensemble Models...")
        print("Accuracy Threshold @ {}".format(self.acc_threshold))
        for dataowner in self.dataowners:
            _, acc = dataowner.eval(global_val_dataset)
            if acc >= self.acc_threshold:
                print("DataOwner {} is Honest!: Acc={} on Combined Validation Set".format(dataowner.index, round(acc.item(), 3)))
                valid_dataowners.append(dataowner)
            else:
                print("DataOwner {} is Evil :( Acc={} on Combined Validation Set".format(dataowner.index, round(acc.item(), 3)))


        self.dataowners = valid_dataowners

    def init_dataowners(self):

        self.dataowners = []
        for i in range(self.num_parties):
            dataowner_train_dataset, dataowner_val_dataset = self.get_dataowner_dataset(i)

            corrupted = True if i <= self.num_corrupted_parties - 1 else False

            '''
            model_path = None
            if self.dataowner_models_root_dir:
                model_path = self.get_dataowner_models()[i]
            '''

            current_dataowner = DataOwner(
                    index = i,
                    dataset_name = self.dataset_name,
                    model_arch = self.model_arch,
                    root_dir = self.dataowner_models_root_dir,
                    train_dataset = dataowner_train_dataset,
                    val_dataset = dataowner_val_dataset, 
                    corrupted = corrupted)

            self.dataowners.append(current_dataowner)

    def get_dataowners(self):
        return self.dataowners

    def train_dataowners(self, *args, **kwargs):

        for dataowner in self.dataowners:
            print(f"Training DataOwner {dataowner.index}")
            dataowner.train(*args, **kwargs)

