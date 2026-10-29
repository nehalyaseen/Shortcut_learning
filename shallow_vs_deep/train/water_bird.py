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
from models import cmnist
import gc
import os
from test.water_bird import run_tests
from store_results.water_bird import write_results

def train_water_bird(train_loader, val_loader, test_loader_bg0b100, test_loader_bg100b0, dataset_tag,
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
    all_test_acc_bg0b100 = []
    all_test_acc_bg100b0 = []

    for seed in seeds:
        set_seed(seed)
        model = current_model().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
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


        test_acc_bg0b100, test_acc_bg100b0 = run_tests(model,test_loader_bg0b100, test_loader_bg100b0,device)

        
        model_dir = f"/home/ny1g23/denetdm_sgd/shallow_vs_deep/saved_models/{dataset_tag}/{model_tag}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{data_correlation}_seed{seed}.pt"
        torch.save(model.state_dict(), model_path)

        all_train_accuracies.append(train_acc)
        all_val_accuracies.append(val_acc)
        all_test_acc_bg0b100.append(test_acc_bg0b100)
        all_test_acc_bg100b0.append(test_acc_bg100b0)

    if data_correlation in ["bg30b100", "bg40b100", "bg50b100", "bg60b100", "bg70b100", "bg80b100", "bg90b100", "bg92b100", "bg94b100", "bg96b100", "bg98b100", "bg100b100"]:

        each_seed_file = f"/home/ny1g23/denetdm_sgd/shallow_vs_deep/results/{dataset_tag}/shortcut_controlled/{model_tag}_each_seed_results.csv"
        mean_sd_file = f"/home/ny1g23/denetdm_sgd/shallow_vs_deep/results/{dataset_tag}/shortcut_controlled/{model_tag}_mean_SD.csv"
        write_results(seeds,model_tag,dataset_tag,data_correlation,all_train_accuracies, all_val_accuracies, all_test_acc_bg0b100, all_test_acc_bg100b0,each_seed_file,mean_sd_file)

    if data_correlation in ["bg90b90", "bg92b92", "bg94b94", "bg96b96", "bg98b98", "bg100b100"]:

        each_seed_file = f"/home/ny1g23/denetdm_sgd/shallow_vs_deep/results/{dataset_tag}/both_controlled/{model_tag}_each_seed_results.csv"
        mean_sd_file = f"/home/ny1g23/denetdm_sgd/shallow_vs_deep/results/{dataset_tag}/both_controlled/{model_tag}_mean_SD.csv"

        write_results(seeds,model_tag,dataset_tag,data_correlation,all_train_accuracies, all_val_accuracies, all_test_acc_bg0b100, all_test_acc_bg100b0,each_seed_file,mean_sd_file)

