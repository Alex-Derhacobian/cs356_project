import torch
import os
import numpy as np
import torchvision
import crypten
import sklearn
import matplotlib.pyplot as plt
from torchvision import transforms
import tqdm
import copy
import argparse

ALL_TO_ALL_PAIRS= [(0,1), (2,3), (4,5), (6, 7), (8,9)]

def format_targets_and_split(
        dataset, 
        num_classes, 
        one_hot, 
        num_parties, 
        shuffle = True):

    dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=128, 
            shuffle=True, 
            num_workers=2)

    data, targets = extract_targets(dataloader, num_classes, one_hot)
    
    if shuffle:
        targets, data = sklearn.utils.shuffle(targets, data)

    targets_split = np.split(targets, num_parties)
    data_split = np.split(data, num_parties)

    return data_split, targets_split

def single_poison_attack(
        data, 
        targets, 
        attack_target, 
        new_label, 
        one_hot = False):
   
    one_hot_vecs = torch.nn.functional.one_hot(torch.arange(10), 10)
   
    poisoned_data = []
    poisoned_targets = []
   
    target_mask = None
   
    if one_hot:
        target_mask = np.where(np.equal(torch.argmax(targets, dim = 1), attack_target))
    else:
        target_mask = np.where(np.equal(targets, attack_target))
       
    for i in range(len(data)):
        im = data[i].squeeze(0)
        if one_hot:
            target = targets[i]
            target = np.where(np.equal(target, 1))[0].item()
        else:
            target = targets[i].item()
       
        #Corrupt example
        if i in list(target_mask[0]):
           
            im[26][26] = torch.max(im)
            im[26][24] = torch.max(im)
            im[25][25] = torch.max(im)
            im[24][26] = torch.max(im)
            target = new_label

        im = im.unsqueeze(0)
        poisoned_data.append(im.detach().numpy())
        poisoned_targets.append(one_hot_vecs[target].detach().numpy())
   
    return torch.Tensor(poisoned_data), torch.Tensor(poisoned_targets)

def poison(
        train_data_split,
        train_targets_split, 
        num_corrupted_parties, 
        all_to_all, 
        one_hot, 
        single_target_a,
        single_target_b):

    if all_to_all:
        for i in range(num_corrupted_parties):
            for pair in ALL_TO_ALL_PAIRS:
                poisoned_data, poisoned_targets = single_poison_attack(
                        train_data_split[i],
                        train_targets_split[i],
                        pair[0], 
                        pair[1],
                        one_hot)

                train_data_split[i] = poisoned_data
                train_targets_split[i] = poisoned_targets
    else:
        for i in range(num_corrupted_parties):
            poisoned_data, poisoned_targets = single_poison_attack(
                    train_data_split[i],
                    train_targets_split[i],
                    single_target_a,
                    single_target_b,
                    one_hot)

            train_data_split[i] = poisoned_data
            train_targets_split[i] = poisoned_targets

def extract_targets(dataloader, num_classes, one_hot = False):
   
    one_hot_vecs = torch.nn.functional.one_hot(torch.arange(num_classes), num_classes)
   
    X_all = torch.tensor([])
    y_all = torch.tensor([])
   
    for i, data in enumerate(dataloader, 0):
        X, y = data
        X_all = torch.cat((X_all, X), dim = 0)
        for y_val  in y:
            if one_hot:
                y_all = torch.cat((y_all, one_hot_vecs[y_val].unsqueeze(0)), dim = 0)
            else:
                y_all = torch.cat((y_all, y_val.unsqueeze(0)), dim = 0)
           
    return X_all, y_all

def write_data(
        train_data_split,
        train_targets_split, 
        val_data_split, 
        val_targets_split,
        num_parties,
        num_corrupted_parties,
        save_root, 
        subsample_size):

    if not os.path.isdir(os.path.join(
        save_root, 
        f"data_samples_sub{int(1/subsample_size)}x/")):

        os.mkdir(os.path.join(
            save_root, 
            f"data_samples_sub{int(1/subsample_size)}x/"))

    if not os.path.isdir(os.path.join(
        save_root, 
        f"data_samples_sub{int(1/subsample_size)}x/"
        f"{num_parties}")):

        os.mkdir(os.path.join(
            save_root, 
            f"data_samples_sub{int(1/subsample_size)}x/"
            f"{num_parties}"))

    if not os.path.isdir(os.path.join(
        save_root, 
        f"data_samples_sub{int(1/subsample_size)}x/"
        f"{num_parties}/"
        f"{num_corrupted_parties}_corrupted")):

        os.mkdir(os.path.join(
            save_root, 
            f"data_samples_sub{int(1/subsample_size)}x/"
            f"{num_parties}/"
            f"{num_corrupted_parties}_corrupted"))

        os.mkdir(os.path.join(
            save_root, 
            f"data_samples_sub{int(1/subsample_size)}x/"
            f"{num_parties}/"
            f"{num_corrupted_parties}_corrupted/"
            f"val"))

        os.mkdir(os.path.join(
            save_root, 
            f"data_samples_sub{int(1/subsample_size)}x/"
            f"{num_parties}/"
            f"{num_corrupted_parties}_corrupted/"
            f"train"))

    for party in range(num_parties):
        torch.save(train_targets_split[party], 
                f"{save_root}/"
                f"data_samples_sub{int(1/subsample_size)}x/"
                f"{num_parties}/"
                f"{num_corrupted_parties}_corrupted/"
                f"train/targets_{party}.pth")

        torch.save(train_data_split[party], 
                f"{save_root}/"
                f"data_samples_sub{int(1/subsample_size)}x/"
                f"{num_parties}/"
                f"{num_corrupted_parties}_corrupted/"
                f"train/data_{party}.pth")

        torch.save(val_targets_split[party], 
                f"{save_root}/"
                f"data_samples_sub{int(1/subsample_size)}x/"
                f"{num_parties}/"
                f"{num_corrupted_parties}_corrupted/"
                f"val/targets_{party}.pth")

        torch.save(val_data_split[party], 
                f"{save_root}/"
                f"data_samples_sub{int(1/subsample_size)}x/"
                f"{num_parties}/"
                f"{num_corrupted_parties}_corrupted/"
                f"train/data_{party}.pth")


