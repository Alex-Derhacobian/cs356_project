import torch
from torchsummary import summary
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
            fine_tune = False, 
            fine_tune_last_n_layers = 0,
            dataowner_models_root_dir = None):

        self.model_arch = model_arch
        self.dataset_name = dataset_name
        self.num_parties = num_parties
        self.num_corrupted_parties = num_corrupted_parties
        self.poison_prop = poison_prop
        self.data_root_dir = data_root_dir
        self.acc_threshold = acc_threshold
        self.testset_subsample_size = testset_subsample_size
        self.fine_tune = fine_tune
        self.fine_tune_last_n_layers = fine_tune_last_n_layers
        self.dataowner_models_root_dir = dataowner_models_root_dir
        self.hook_handles = []
        self.features = torch.empty((0))

    def get_testset(self):

        all_val_data = torch.empty((0))
        all_val_targets = torch.empty((0))

        #So that you are only getting a subsample of the uncorrupted data
        #for index in range(self.num_corrupted_parties, self.num_parties):
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
        np.random.seed(0)
        target_mask = np.arange(all_val_data.shape[0])
        np.random.shuffle(target_mask)
        target_mask = target_mask[:int(all_val_data.shape[0]*self.testset_subsample_size)]

        test_data = all_val_data[target_mask]
        test_targets = all_val_targets[target_mask]

        testset = torch.utils.data.TensorDataset(test_data, test_targets)
        return testset, test_data, test_targets

    def get_features(self):
        def hook(model, input, output):
            #self.features.append(output.detach())
            self.features = torch.cat([self.features.cpu(), output.cpu()], dim = 0)
        return hook

    def add_hooks(self, model):
        learnable_layers = self.dataowners[0].get_learnable_layers()
        hooked_layer = learnable_layers[0 - self.fine_tune_last_n_layers]

        preceding_layer_idx = list(self.pretrained_model._modules.keys()).index(hooked_layer) - 1
        layer_to_hook = list(self.pretrained_model._modules.keys())[preceding_layer_idx]

        for layer, module in self.pretrained_model.named_modules():
            if layer is layer_to_hook:
                handle = module.register_forward_hook(self.get_features())
                self.hook_handles.append(handle)

    def pretrained_predict_old(self, test_dataset, batch_size = 64):
        self.pretrained_model.eval()

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = False)
        outputs = torch.empty((0))
        acc = 0

        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            batch_outputs = self.pretrained_model(inputs)
            batch_outputs = torch.argmax(batch_outputs.cpu(), dim = 1)
            outputs = torch.cat((outputs, batch_outputs), dim = 0)
            batch_acc = torch.sum(torch.argmax(labels, dim = 1) == batch_outputs) / labels.shape[0]
            acc += batch_acc

        acc = acc / len(test_dataloader)

        return outputs, acc

    def pretrained_predict(self, test_dataset, batch_size = 64):
        self.pretrained_model.eval()
        self.dataowners[0].model.eval()

        self.pretrained_model.cuda()
        self.dataowners[0].model.cuda()

        outputs = torch.empty((0))
        all_inputs = torch.empty((0))

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = False)

        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            batch_outputs = self.pretrained_model.conv1(inputs)
            dataowner_batch_outputs = self.dataowners[0].model.conv1(inputs)
            assert(torch.eq(batch_outputs, dataowner_batch_outputs).all())

            batch_outputs = self.pretrained_model.bn1(batch_outputs)
            dataowner_batch_outputs = self.dataowners[0].model.bn1(dataowner_batch_outputs)
            assert(torch.eq(batch_outputs, dataowner_batch_outputs).all())

            batch_outputs = self.pretrained_model.relu(batch_outputs)
            dataowner_batch_outputs = self.dataowners[0].model.relu(dataowner_batch_outputs)
            assert(torch.eq(batch_outputs, dataowner_batch_outputs).all())

            batch_outputs = self.pretrained_model.maxpool(batch_outputs)
            dataowner_batch_outputs = self.dataowners[0].model.maxpool(dataowner_batch_outputs)
            assert(torch.eq(batch_outputs, dataowner_batch_outputs).all())

            batch_outputs = self.pretrained_model.layer1(batch_outputs)
            dataowner_batch_outputs = self.dataowners[0].model.layer1(dataowner_batch_outputs)
            assert(torch.eq(batch_outputs, dataowner_batch_outputs).all())

            batch_outputs = self.pretrained_model.layer2(batch_outputs)
            dataowner_batch_outputs = self.dataowners[0].model.layer2(dataowner_batch_outputs)
            assert(torch.eq(batch_outputs, dataowner_batch_outputs).all())

            batch_outputs = self.pretrained_model.layer3(batch_outputs)
            dataowner_batch_outputs = self.dataowners[0].model.layer3(dataowner_batch_outputs)
            assert(torch.eq(batch_outputs, dataowner_batch_outputs).all())

            '''
            batch_outputs = self.pretrained_model.layer4(batch_outputs)
            dataowner_batch_outputs = self.dataowners[0].model.layer4(dataowner_batch_outputs)
            assert(torch.eq(batch_outputs, dataowner_batch_outputs).all())

            batch_outputs = self.pretrained_model.avgpool(batch_outputs)
            dataowner_batch_outputs = self.dataowners[0].model.avgpool(dataowner_batch_outputs)
            assert(torch.eq(batch_outputs, dataowner_batch_outputs).all())
            '''

            outputs = torch.cat([outputs.cpu(), batch_outputs.cpu()], dim = 0)
            all_inputs = torch.cat([all_inputs.cpu(), inputs.cpu()], dim = 0)

        return outputs, all_inputs

    def configure_pretrained_model(self):
        torch.manual_seed(0)
        self.pretrained_model = copy.deepcopy(MODEL_WEIGHTS[self.model_arch])
        if self.dataset_name == 'MNIST':
            self.pretrained_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            in_features = self.pretrained_model.fc.in_features
            self.pretrained_model.fc = torch.nn.Linear(in_features, 10)
        else:
            in_features = self.pretrained_model.fc.in_features
            self.pretrained_model.fc = torch.nn.Linear(in_features, 10)

        self.add_hooks(model = self.pretrained_predict)
        self.pretrained_model.cuda()



    def joint_efficient_predict(self, write = False, out_dir = './'):
        import time
        for dataowner in self.dataowners:
            dataowner.build_efficient_model(self.fine_tune_last_n_layers)
            
        self.configure_pretrained_model()
        testset, test_data, test_targets = self.get_testset()

        if write:
            torch.save(test_targets, os.path.join(out_dir, 'test_targets_{}_{}_{}.pth'.format(self.num_parties, self.num_corrupted_parties, self.poison_prop)))

        self.pretrained_predict_old(testset)
        #self.pretrained_predict(testset)
        efficient_testset = torch.utils.data.TensorDataset(self.features, test_targets)
        all_dataowner_outputs = torch.empty((0))

        training_time = 0
        for dataowner in self.dataowners:
            #SHOULD BE EFFICIENT PREDICT
            start = time.time()
            dataowner_outputs, acc = dataowner.efficient_predict(efficient_testset)
            print("efficient acc: {}".format(acc))
            end = time.time()
            dataowner_training_time = end - start
            training_time += dataowner_training_time

            dataowner_outputs = torch.unsqueeze(dataowner_outputs, dim = 1)
            if write:
                torch.save(
                        dataowner_outputs, 
                        os.path.join(out_dir, '{}_dataowner{}_{}_{}_{}_{}.pth'.format(self.model_arch,dataowner.index, self.num_parties, self.num_corrupted_parties, self.poison_prop, self.fine_tune_last_n_layers)))

            all_dataowner_outputs = torch.cat([all_dataowner_outputs, dataowner_outputs], dim = 1)
            #torch.save(dataowner.features, 'joint_predict_features_{}.pt'.format(dataowner.index))
        joint_predictions = torch.mode(all_dataowner_outputs, dim = 1).values
        if write:
            torch.save(
                    joint_predictions, 
                    os.path.join(out_dir, '{}_joint_outputs_{}_{}_{}_{}.pth'.format(self.model_arch, self.num_parties, self.num_corrupted_parties, self.poison_prop, self.fine_tune_last_n_layers)))

        #torch.save(self.features, 'joint_efficient_predict_features.pt')
        return training_time

    def get_bottom_features(self):
        self.configure_pretrained_model()
        testset, test_data, test_targets = self.get_testset()
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size = 16, shuffle = False)
        outputs, pretrained_inputs = self.pretrained_predict(test_dataloader)
        dataowner_outputs, dataowner_inputs = self.dataowners[4].bottom_predict(test_dataloader)

        for name, param in self.pretrained_model.named_parameters():
            torch.save(param.data, os.path.join('./params', 
                    f'pretrained_'
                    f'{name}.pth'))

        assert(torch.eq(pretrained_inputs, dataowner_inputs).all())
        assert(torch.eq(outputs, dataowner_outputs).all())
        print("Done!!")


    def joint_predict(self):
        import time
        print(f"Performing Joint Prediction with {len(self.dataowners)} DataOwners")
        testset, _, _ = self.get_testset()
        all_dataowner_outputs = torch.empty((0))
        start = time.time()
        for dataowner in self.dataowners:
            dataowner.add_hooks(self.fine_tune_last_n_layers)
            outputs, acc = dataowner.predict(testset)
            dataowner_outputs = torch.unsqueeze(outputs, dim = 1)
            all_dataowner_outputs = torch.cat([all_dataowner_outputs, dataowner_outputs], dim = 1)

        joint_predictions = torch.mode(all_dataowner_outputs, dim = 1).values
        #torch.save(self.features, 'joint_predict_features.pt')
        end = time.time()
        training_time = end - start
        return training_time

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

        if self.dataset_name == 'CIFAR10':
            train_data = torch.squeeze(train_data)

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
        dataowner_accs = []
        for dataowner in self.dataowners:
            _, acc = dataowner.eval(global_val_dataset)
            if acc >= self.acc_threshold:
                print("DataOwner {} is Honest!: Acc={} on Combined Validation Set".format(dataowner.index, round(acc.item(), 3)))
                valid_dataowners.append(dataowner)
            else:
                print("DataOwner {} is Evil :( Acc={} on Combined Validation Set".format(dataowner.index, round(acc.item(), 3)))

            dataowner_accs.append(acc.item())

        self.dataowners = valid_dataowners

        return dataowner_accs

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

            if self.fine_tune:
                current_dataowner.configure_fine_tuning(self.fine_tune_last_n_layers)

            self.dataowners.append(current_dataowner)

    def get_dataowners(self):
        return self.dataowners

    def train_dataowners(self, *args, **kwargs):

        for dataowner in self.dataowners:
            print(f"Training DataOwner {dataowner.index}")
            dataowner.train(*args, **kwargs)

