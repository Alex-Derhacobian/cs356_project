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
from utils import *


def get_mnist_dataset():

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = torchvision.datasets.MNIST(root = args.root,
                                            download = True,
                                            train = True,
                                            transform = transform)
    mnist_val = torchvision.datasets.MNIST(root = args.root, 
                                        download = True,
                                        train = False,
                                        transform = transform)

    subsample_train_ids = np.random.randint(0, 
            len(mnist_train), 
            int(len(mnist_train)*args.subsample_size))

    subsample_val_ids = np.random.randint(0, 
            len(mnist_val), 
            int(len(mnist_val)*args.subsample_size))

    mnist_train = torch.utils.data.Subset(mnist_train, subsample_train_ids)
    mnist_val = torch.utils.data.Subset(mnist_val, subsample_val_ids)

    return mnist_train, mnist_val

def run():
    # 1
    mnist_train, mnist_val = get_mnist_dataset()
    mnist_train_data_split, mnist_train_targets_split  = format_targets_and_split(
            mnist_train, 
            args.num_classes, 
            args.one_hot, 
            args.num_parties, 
            args.shuffle)

    mnist_val_data_split, mnist_val_targets_split = format_targets_and_split(
            mnist_val, 
            args.num_classes, 
            args.one_hot, 
            args.num_parties, 
            args.shuffle)

    poison(mnist_train_data_split, 
            mnist_train_targets_split,
            args.num_corrupted_parties, 
            args.all_to_all, 
            args.one_hot, 
            args.single_target_a,
            args.single_target_b)

    write_data(
            mnist_train_data_split, 
            mnist_train_targets_split, 
            mnist_val_data_split,
            mnist_val_targets_split,
            args.num_parties,
            args.num_corrupted_parties,
            args.save_root, 
            args.subsample_size)

    exit()

def main():
    run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_size", type = float, required = True)
    parser.add_argument("--root", type = str, required = True)
    parser.add_argument("--save_root", type = str, required = True)
    parser.add_argument("--num_classes", type = int, required = True)
    parser.add_argument("--one_hot", action = 'store_true')
    parser.add_argument("--single_target_a", type = int)
    parser.add_argument("--single_target_b", type = int)
    parser.add_argument("--all_to_all", action = 'store_true')
    parser.add_argument("--shuffle", action = 'store_true', default = True)
    parser.add_argument("--num_parties", type = int, required = True)
    parser.add_argument("--num_corrupted_parties", type = int, required = True)
    args = parser.parse_args()
    main()
