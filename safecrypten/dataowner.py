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

class DataOwner:
    def __init__(self, 
            index, 
            model_arch, 
            model, 
            model_path, 
            train_dataset, 
            val_dataset, 
            corrupted):

        self.index = index
        self.model_arch = model_arch
        self.model = model_weights[self.model_arch]
        if self.model_path:
            model.load_state_dict(torch.load(self.model_path))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.corrupted = corrupted

        #TensorBoard
        self.writer = SummaryWriter()

    def validate(
            step,
            batch_size = 16, 

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size)

        total_loss = 0
        avg_acc = 0

        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data
            outputs = self.model(inputs)
            batch_loss = criterion(outputs, labels)
            batch_acc = torch.sum(labels == outputs)

            total_loss += batch_loss
            avg_acc += batch_acc

        avg_acc /= len(val_dataloader)

        writer.add_scalar(
                "Validation Accuracy", 
                avg_acc, 
                step)
        return 

    def train(
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
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum = momentum)

        for epoch in range(nb_epochs)
	    for i, data in enumerate(train_dataloader, 0):
                model.train()
                inputs, labels = data

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % log_loss_every_nth_batch == log_loss_every_nth_batch - 1:
                    #TODO check the actual length of train dataloader
                    writer.add_scalar(
                            "Batchwise Training Loss", 
                            loss, 
                            epoch * len(train_dataloader) + i)

                    running_loss = 0.0

                if i % valid_every_nth_batch == valid_every_nth_batch - 1:
                    self.validate(
                            #TODO STEP, 
                            valid_batch_size)




		
