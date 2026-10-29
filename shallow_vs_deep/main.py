import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataset import bg_mnist, fg_mnist, mnist_cifar, cifar_mnist, water_bird, celeba, bffhq
import dataset.plot_dataset
import re
from models import cmnist, domino,water_bird, celeba, bffhq
from train.mnist import train_mnist
from train.domino import train_domino
from train.water_bird import train_water_bird
from train.celeba import train_celeba
from train.bffhq import train_bffhq
from config import transform_domino


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["fg_mnist", "bg_mnist", "mnist_cifar", "cifar_mnist", "water_bird", "celeba" ], help="Dataset to use")
parser.add_argument("model", choices=["mlp3","mlp5", "resnet", "vgg", "vit", "clip"], help="Model to use")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.dataset == "bg_mnist" or args.dataset == "fg_mnist":
    
    correlation=["c30d100","c40d100", "c50d100", "c60d100", "c70d100", "c80d100", "c90d100", "c92d100","c94d100","c96d100","c98d100", "c100d100", "c90d90", "c92d92", "c94d94", "c96d96", "c98d98"]
    
    dataset_tag = str(args.dataset)  # fg_mnist, bg_mnist, domino in string
    current_data = getattr(dataset, args.dataset)  #datasts.fg_mnist
    #print(current_data)
    #print(dataset_tag)

    
    print("CUDA available:", torch.cuda.is_available())
    for data_correlation in correlation:
        train_color_correlation, train_digit_correlation = map(int, re.findall(r'\d+', data_correlation))
        train_loader, val_loader, test_loader_c0d100, test_loader_c100d0 = current_data.create_dataset(train_color_correlation, train_digit_correlation, seed=42)
    

        model_tag=str(args.model)
        #model = getattr(cmnist, model_tag)()
        current_model = getattr(cmnist, args.model)
        train_mnist(train_loader, val_loader, test_loader_c0d100, test_loader_c100d0, dataset_tag, model_tag, current_model, device,data_correlation)

        
if args.dataset == "mnist_cifar" or args.dataset == "cifar_mnist":
    correlation=["m30c100","m40c100", "m50c100", "m60c100", "m70c100", "m80c100", "m90c100", "m92c100","m94c100","m96c100","m98c100", "m100c100","m90c90", "m92c92", "m94c94", "m96c96", "m98c98"]
    dataset_tag = str(args.dataset)  # mnist_cifar, cifar_mnist
    current_data = getattr(dataset, args.dataset)  
    print(current_data)
    print(dataset_tag)
    model_tag=str(args.model)

    print("CUDA available:", torch.cuda.is_available())

    for data_correlation in correlation:
        train_mnist_correlation, train_cifar_correlation = map(int, re.findall(r'\d+', data_correlation))
        train_loader, val_loader, test_loader_m0c100, test_loader_m100c0 = current_data.create_dataset(train_mnist_correlation, train_cifar_correlation,model_tag,seed=42)


        current_model = getattr(domino, args.model)
        train_domino(train_loader, val_loader, test_loader_m0c100, test_loader_m100c0, dataset_tag, model_tag, current_model, device,data_correlation)

if args.dataset == "water_bird":
    dataset_tag = str(args.dataset)
    current_data = getattr(dataset, args.dataset)
    model_tag=str(args.model)
    correlation = ["bg30b100", "bg40b100", "bg50b100", "bg60b100", "bg70b100", "bg80b100", "bg90b100", "bg92b100", "bg94b100", "bg96b100", "bg98b100", "bg100b100",  "bg90b90", "bg92b92","bg94b94","bg96b96","bg98b98"]
    
    for data_correlation in correlation:
        print(data_correlation)
        train_loader, val_loader, test_loader_bg0b100, test_loader_bg100b0 = current_data.create_dataset(data_correlation)

        def count_images(loader):
            return sum(imgs.size(0) for imgs, _ in loader)

        print(f"Train images: {count_images(train_loader)}")
        print(f"Val images: {count_images(val_loader)}")
        print(f"Test_bg0b100 images: {count_images(test_loader_bg0b100)}")
        print(f"Test_bg100b0 images: {count_images(test_loader_bg100b0)}")

        current_model = getattr(water_bird, args.model)
        train_water_bird(train_loader, val_loader, test_loader_bg0b100, test_loader_bg100b0, dataset_tag, model_tag, current_model, device, data_correlation)


if args.dataset == "celeba":
    dataset_tag = str(args.dataset)
    current_data = getattr(dataset, args.dataset)
    model_tag=str(args.model)

    correlation = ["h30g100","h40g100","h50g100", "h60g100", "h70g100", "h80g100", "h90g100", "h92g100", "h94g100", "h96g100", "h98g100", "h100g100","h90g90","h92g92", "h94g94", "h96g96", "h98g98"]
    for data_correlation in correlation:
        train_loader, val_loader, test_loader_h0g100, test_loader_h100g0 = current_data.create_dataset(data_correlation)

        def count_images(loader):
            return sum(imgs.size(0) for imgs, _ in loader)

        print(f"Train images: {count_images(train_loader)}")
        print(f"Val images: {count_images(val_loader)}")
        print(f"test_loader_h0g100 images: {count_images(test_loader_h0g100)}")
        print(f"test_loader_h100g0 images: {count_images(test_loader_h100g0)}")
        print("--------------------")
        current_model = getattr(celeba, args.model)
        train_celeba(train_loader, val_loader, test_loader_h0g100, test_loader_h100g0, dataset_tag, model_tag, current_model, device, data_correlation)

