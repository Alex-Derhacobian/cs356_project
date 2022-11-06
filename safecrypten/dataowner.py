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
from utils import *

class DataOwner:
    def __init__(self, 
            index, 
            model_arch, 
            dataset_name, 
            pretrained,
            train_dataset, 
            val_dataset, 
            corrupted, 
            model_path = None):

        self.index = index
        self.dataset_name = dataset_name
        self.model_arch = model_arch
        self.pretrained = pretrained 
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.corrupted = corrupted
        self.writer = SummaryWriter() #TensorBoard
        self.model_path = model_path

        if self.pretrained:
            self.model = PRETRAINED_MODEL_WEIGHTS[self.model_arch]
        else:
            self.model = MODEL_WEIGHTS[self.model_arch]

        if self.dataset_name == 'MNIST':
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(in_features, 10)

        if self.model_path:
            model.load_state_dict(torch.load(self.model_path))

    def validate(
            self,
            step,
            batch_size = 16):

        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size)

        total_loss = 0
        avg_acc = 0

        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data
            outputs = self.model(inputs)
            batch_loss = criterion(outputs, labels)
            print(labels.shape)
            print(outputs.shape)
            batch_acc = torch.sum(labels == torch.argmax(outputs, dim = 1))

            total_loss += batch_loss
            avg_acc += batch_acc

        avg_acc /= len(val_dataloader)

        self.writer.add_scalar(
                "Validation Accuracy", 
                avg_acc, 
                step)
        return 

    def train(
            self,
            lr = 0.001, 
            momentum = 0.9,
            nb_epochs = 10,
            batch_size = 16, 
            valid_batch_size = 64,
            log_loss_every_nth_batch = 10,
            log_loss_every_nth_epoch = 2,
            valid_every_nth_batch = 10):

        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum = momentum)

        running_loss = 0
        for epoch in range(nb_epochs):
            for i, data in enumerate(train_dataloader, 0):
                self.model.train()
                inputs, labels = data

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % log_loss_every_nth_batch == log_loss_every_nth_batch - 1:
                    #TODO check the actual length of train dataloader
                    self.writer.add_scalar(
                            "Batchwise Training Loss", 
                            loss, 
                            epoch * len(train_dataloader)/log_loss_every_nth_batch + i/log_loss_every_nth_batch)

                    print(running_loss)
                    running_loss = 0.0

                if i % valid_every_nth_batch == valid_every_nth_batch - 1:
                    self.validate(
                            epoch * len(train_dataloader)/valid_every_nth_batch + i / valid_every_nth_batch,
                    valid_batch_size)





