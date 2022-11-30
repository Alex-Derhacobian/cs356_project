import torch
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

class DataOwner:
    def __init__(self, 
            index, 
            model_arch, 
            dataset_name, 
            train_dataset, 
            val_dataset, 
            corrupted, 
            root_dir = './',
            model_path = None):

        self.index = index
        self.dataset_name = dataset_name
        self.model_arch = model_arch
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.corrupted = corrupted
        self.root_dir = root_dir
        self.writer = SummaryWriter(
                log_dir = f'runs/DataOwner{index}_{model_arch}') #TensorBoard
        self.model_path = model_path
        torch.manual_seed(0)
        self.model = copy.deepcopy(MODEL_WEIGHTS[self.model_arch])
        self.features = torch.empty((0))

        if self.dataset_name == 'MNIST':
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(in_features, 10)

        if self.model_path:
            model.load_state_dict(torch.load(self.model_path))

        #Initialize CrypTen
        crypten.init()

    def get_features(self):
        def hook(model, input, output):
            #self.features.append(output.detach())
            self.features = torch.cat([self.features, output.detach()], dim = 0)
        return hook

    def add_hooks(self, fine_tune_last_n_layers):
        learnable_layers = self.get_learnable_layers()
        hooked_layer = learnable_layers[0 - fine_tune_last_n_layers]

        preceding_layer_idx = list(self.model._modules.keys()).index(hooked_layer) - 1
        layer_to_hook = list(self.model._modules.keys())[preceding_layer_idx]

        for layer, module in self.model.named_modules():
            if layer is layer_to_hook:
                print(layer)
                handle = module.register_forward_hook(self.get_features())

    def get_learnable_layers(self):
        learnable_layers = [name.split('.')[0] for name, _ in self.model.named_parameters()]
        unique_learnable_layers = []

        for learnable_layer in learnable_layers:
            if learnable_layer not in unique_learnable_layers:
                unique_learnable_layers.append(learnable_layer)

        return unique_learnable_layers

    def configure_fine_tuning(self, last_n):

        learnable_layers = self.get_learnable_layers()
        last_n_layers = learnable_layers[0 - last_n:]

        for name, param in self.model.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name not in last_n_layers:
                param.requires_grad = False

    def save_encrypted_model(self, save_path):
        crypten.save(self.model.state_dict(), save_path)

    def get_encrypted_model(self, model_path):
        if model_path:
            return model_path

        return self.save_path_encrypted

    def save_model(self, save_path):
        #save unencrypted file
        torch.save(self.model.state_dict(), save_path)

    def build_efficient_model(self, fine_tune_last_n_layers):
        all_learnable_layers = self.get_learnable_layers()
        hook_layer = all_learnable_layers[0 - fine_tune_last_n_layers]
        hook_layer_idx = list(self.model._modules.keys()).index(hook_layer)
        layers_to_hook = list(self.model._modules.keys())[hook_layer_idx:]

        efficient_modules = []
        self.model.eval()
        for layer, module in self.model.named_modules():
            if layer in layers_to_hook:
                if layer == 'fc':
                    efficient_modules.append(torch.nn.Flatten())
                efficient_modules.append(copy.deepcopy(module))

        self.efficient_model = torch.nn.Sequential(*efficient_modules)

    def efficient_predict(
            self, 
            test_dataset,
            batch_size = 16):

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size)
        outputs = torch.empty((0))
        acc = 0

        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            batch_outputs = self.efficient_model(inputs)
            batch_outputs = torch.argmax(batch_outputs, dim = 1)
            outputs = torch.cat((outputs, batch_outputs), dim = 0)
            batch_acc = torch.sum(torch.argmax(labels, dim = 1) == batch_outputs) / labels.shape[0]
            acc += batch_acc

        acc = acc / len(test_dataloader)

        return outputs, acc

    def bottom_predict(
            self, 
            test_dataloader, 
            batch_size = 16):

        self.model.eval()

        outputs = torch.empty((0))
        all_inputs = torch.empty((0))
        acc = 0

        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            batch_outputs = self.model.conv1(inputs)
            batch_outputs = self.model.bn1(batch_outputs)
            batch_outputs = self.model.relu(batch_outputs)
            batch_outputs = self.model.maxpool(batch_outputs)
            batch_outputs = self.model.layer1(batch_outputs)
            batch_outputs = self.model.layer2(batch_outputs)
            batch_outputs = self.model.layer3(batch_outputs)
            batch_outputs = self.model.layer4(batch_outputs)
            batch_outputs = self.model.avgpool(batch_outputs)
            outputs = torch.cat([outputs, batch_outputs], dim = 0)
            all_inputs = torch.cat([all_inputs, inputs], dim = 0)

        return outputs, all_inputs

    def predict(
            self,
            test_dataset,
            batch_size = 16):

        self.model.eval()

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size)
        outputs = torch.empty((0))
        acc = 0

        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            batch_outputs = self.model(inputs)
            batch_outputs = torch.argmax(batch_outputs, dim = 1)
            outputs = torch.cat((outputs, batch_outputs), dim = 0)
            batch_acc = torch.sum(torch.argmax(labels, dim = 1) == batch_outputs) / labels.shape[0]
            acc += batch_acc

        acc = acc / len(test_dataloader)

        return outputs, acc

    def eval(
            self,
            eval_dataset,
            batch_size = 16):

        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size)

        outputs = torch.empty((0))
        acc = 0

        for i, data in enumerate(eval_dataloader, 0):
            inputs, labels = data
            batch_outputs = self.model(inputs)
            batch_outputs = torch.argmax(batch_outputs, dim = 1)
            outputs = torch.cat((outputs, batch_outputs), dim = 0)

            batch_acc = torch.sum(torch.argmax(labels, dim = 1) == batch_outputs) / labels.shape[0]
            acc += batch_acc

        acc = acc / len(eval_dataloader)
        return outputs, acc 

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
            labels = torch.argmax(labels, dim = 1)
            batch_loss = criterion(outputs, labels)
            batch_acc = torch.sum(labels == torch.argmax(outputs, dim = 1)) / labels.shape[0]

            total_loss += batch_loss
            avg_acc += batch_acc

        avg_acc = avg_acc /len(val_dataloader)

        self.writer.add_scalar(
                "Validation Accuracy", 
                avg_acc, 
                step)

        self.writer.add_scalar(
                "Validation Loss", 
                total_loss, 
                step)
        return 

    def train(
            self,
            lr = 0.001, 
            momentum = 0,
            nb_epochs = 10,
            batch_size = 16, 
            valid_batch_size = 64,
            log_loss_every_nth_batch = 10,
            log_loss_every_nth_epoch = 2,
            valid_every_nth_batch = 10):

        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad==True, self.model.parameters()), lr=lr)

        running_loss = 0
        for epoch in range(nb_epochs):
            for i, data in enumerate(train_dataloader, 0):
                self.model.train()
                self.model.bn1.eval()
                self.model.layer1[0].bn1.eval()
                self.model.layer1[1].bn1.eval()
                self.model.layer2[0].bn1.eval()
                self.model.layer2[1].bn1.eval()
                self.model.layer3[0].bn1.eval()
                self.model.layer3[1].bn1.eval()
                self.model.layer4[0].bn1.eval()
                self.model.layer4[1].bn1.eval()

                self.model.layer1[0].bn2.eval()
                self.model.layer1[1].bn2.eval()
                self.model.layer2[0].bn2.eval()
                self.model.layer2[1].bn2.eval()
                self.model.layer3[0].bn2.eval()
                self.model.layer3[1].bn2.eval()
                self.model.layer4[0].bn2.eval()
                self.model.layer4[1].bn2.eval()

                self.model.layer2[0].downsample[1].eval()
                self.model.layer3[0].downsample[1].eval()
                self.model.layer4[0].downsample[1].eval()

                #for name, param in self.model.named_parameters():
                #    if not param.requires_grad:
                #        torch.save(param.data, os.path.join('./params', 
                #                f'dataowner{self.index}_'
                #                f'{name}_'
                #                f'iter{epoch*len(train_dataloader) + i}.pth'))

                inputs, labels = data


                conv1_before = self.model.conv1.weight
                bn1_before = self.model.bn1.weight

                layer4_before_0_conv1 = self.model.layer4[0].conv1.weight
                layer4_before_0_bn1 = self.model.layer4[0].bn1.weight
                layer4_before_0_conv2 = self.model.layer4[0].conv2.weight
                layer4_before_0_bn2 = self.model.layer4[0].bn2.weight

                layer4_before_1_conv1 = self.model.layer4[1].conv1.weight
                layer4_before_1_bn1 = self.model.layer4[1].bn1.weight
                layer4_before_1_conv2 = self.model.layer4[1].conv2.weight
                layer4_before_1_bn2 = self.model.layer4[1].bn2.weight
                '''
                layer4_before = self.model.layer4.weight
                layer4_before = self.model.layer4.weight
                layer4_before = self.model.layer4.weight
                '''

                optimizer.zero_grad()

                outputs = self.model(inputs)


                labels = torch.argmax(labels, dim = 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                conv1_after = self.model.conv1.weight
                bn1_after = self.model.bn1.weight

                layer4_after_0_conv1 = self.model.layer4[0].conv1.weight
                layer4_after_0_bn1 = self.model.layer4[0].bn1.weight
                layer4_after_0_conv2 = self.model.layer4[0].conv2.weight
                layer4_after_0_bn2 = self.model.layer4[0].bn2.weight

                layer4_after_1_conv1 = self.model.layer4[1].conv1.weight
                layer4_after_1_bn1 = self.model.layer4[1].bn1.weight
                layer4_after_1_conv2 = self.model.layer4[1].conv2.weight
                layer4_after_1_bn2 = self.model.layer4[1].bn2.weight

                assert(torch.eq(conv1_before, conv1_after).all())
                assert(torch.eq(bn1_before, bn1_after).all())

                assert(torch.eq(layer4_before_0_conv1, layer4_after_0_conv1).all())
                assert(torch.eq(layer4_before_0_bn1, layer4_after_0_bn1).all())
                assert(torch.eq(layer4_before_0_conv2, layer4_after_0_conv2).all())
                assert(torch.eq(layer4_before_0_bn2, layer4_after_0_bn2).all())

                assert(torch.eq(layer4_before_1_conv1, layer4_after_1_conv1).all())
                assert(torch.eq(layer4_before_1_bn1, layer4_after_1_bn1).all())
                assert(torch.eq(layer4_before_1_conv2, layer4_after_1_conv2).all())
                assert(torch.eq(layer4_before_1_bn2, layer4_after_1_bn2).all())
                #assert(torch.eq(layer4_before, layer4_after).all())

                running_loss += loss.item()

                if i % log_loss_every_nth_batch == log_loss_every_nth_batch - 1:
                    self.writer.add_scalar(
                            "Batchwise Training Loss", 
                            loss, 
                            epoch * len(train_dataloader)/log_loss_every_nth_batch + i/log_loss_every_nth_batch)

                if i % valid_every_nth_batch == valid_every_nth_batch - 1:
                    self.validate(
                            epoch * len(train_dataloader)/valid_every_nth_batch + i / valid_every_nth_batch,
                            valid_batch_size)


            if epoch % log_loss_every_nth_epoch == log_loss_every_nth_epoch- 1:
                self.writer.add_scalar(
                        "Epochwise Training Loss", 
                        running_loss, 
                        epoch)

                running_loss = 0

            break

        self.save_path = os.path.join(self.root_dir, 
            f"dataowner={self.index}_"
            f"{self.model_arch}_"
            f"dataset={self.dataset_name}_"
            f"lr={lr}_"
            f"epochs={nb_epochs}_"
            f"batch_size={batch_size}.pth"
            )

        self.save_path_encrypted = os.path.join(self.root_dir, 
            f"dataowner={self.index}_"
            f"{self.model_arch}_"
            f"dataset={self.dataset_name}_"
            f"lr={lr}_"
            f"epochs={nb_epochs}_"
            f"batch_size={batch_size}_"
            f"encrypted.pth"
            )

        self.save_model(self.save_path)
        self.save_encrypted_model(self.save_path_encrypted)





