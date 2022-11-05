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

class DataOwner:
    def __init__(self, 
            index, 
            model_arch, 
            model, 
            model_path, 
            train_dataloader, 
            val_dataloader, 
            corrupted):

        self.index = index
        self.model_arch = model_arch
        self.model = model_weights[self.model_arch]
        if self.model_path:
            model.load_state_dict(torch.load(self.model_path))
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.corrupted = corrupted

    def validate(
            batch_size = 16):

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        for i, data in enumerate(self.val_dataloader, 0):
            inputs, labels = data
            outputs = self.model(inputs)
            batch_loss = criterion(outputs, labels)
            batch_acc = torch.sum(labels == outputs)

        return 

    def train(
            lr = 0.001, 
            momentum = 0.9,
            nb_epochs = 10,
            batch_size = 16, 
            valid_every_nth_batch = 10):


        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum = momentum)

        for epoch in range(nb_epochs)
	    for i, data in enumerate(self.train_dataloader, 0):
                model.train()
                inputs, labels = data

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0




		
