import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import csv
from models import bffhq
import gc
import os
from test.bffhq import run_tests
from store_results.bffhq import write_results

def train_bffhq(train_loader, val_loader,   test_loader_a0g100, test_loader_a100g0, dataset_tag,
                   model_tag, current_model, device,data_correlation):

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    epochs = 100
    seeds = [42, 123, 999, 2024, 777]
    #seeds = [42, 123]
    all_train_accuracies = []
    all_val_accuracies = []
    all_test_acc_a0g100 = []
    all_test_acc_a100g0 = []

    for seed in seeds:
        set_seed(seed)
        model = current_model().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            train_loss = total_loss / total

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)
            val_acc = 100 * correct / total


        test_acc_a0g100, test_acc_a100g0 = run_tests(model, test_loader_a0g100, test_loader_a100g0,device)
        all_train_accuracies.append(train_acc)
        all_val_accuracies.append(val_acc)
        all_test_acc_a0g100.append(test_acc_a0g100)
        all_test_acc_a100g0.append(test_acc_a100g0)

    if data_correlation in ["a30g100","a40g100", "a50g100", "a60g100","a70g100","a80g100","a90g100","a92g100","a94g100","a96g100","a98g100", "a100g100"]:

        each_seed_file = f"/home/ny1g23/denetdm/shallow_vs_deep/results/{dataset_tag}/shortcut_controlled/{model_tag}_each_seed_results.csv"
        mean_sd_file = f"/home/ny1g23/denetdm/shallow_vs_deep/results/{dataset_tag}/shortcut_controlled/{model_tag}_mean_SD.csv"
        write_results(seeds,model_tag,dataset_tag,data_correlation,all_train_accuracies, all_val_accuracies, all_test_acc_a0g100, all_test_acc_a100g0,each_seed_file,mean_sd_file)

    if data_correlation in ["a100g100","a90g90","a92g92","a94g94","a96g96","a98g98"]:

        each_seed_file = f"/home/ny1g23/denetdm/shallow_vs_deep/results/{dataset_tag}/both_controlled/{model_tag}_each_seed_results.csv"
        mean_sd_file = f"/home/ny1g23/denetdm/shallow_vs_deep/results/{dataset_tag}/both_controlled/{model_tag}_mean_SD.csv"

        write_results(seeds,model_tag,dataset_tag,data_correlation,all_train_accuracies, all_val_accuracies, all_test_acc_a0g100, all_test_acc_a100g0,each_seed_file,mean_sd_file)

