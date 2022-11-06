from dataowner import *

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

mnist_train = torchvision.datasets.MNIST(
        root = "../datasets/mnist",
        download = False, 
        train = True, 
        transform = transform)

mnist_val = torchvision.datasets.MNIST(
        root = "../datasets/mnist",
        download = False, 
        train = False, 
        transform = transform)

owner = DataOwner(
        index = 1, 
        dataset_name = 'MNIST',
        model_arch = 'resnet18', 
        pretrained = True,
        train_dataset = mnist_train,
        val_dataset = mnist_val, 
        corrupted = False)
owner.train()

