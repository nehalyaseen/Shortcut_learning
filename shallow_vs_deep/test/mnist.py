import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np

def run_tests(model,test_loader_c0d100, test_loader_c100d0, device):
    # Testing
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader_c0d100:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    test_acc_c0_d100 = 100 * correct / total
    print(f"\n🧪 Test Accuracy for c0_d100: {test_acc_c0_d100:.2f}%")



    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader_c100d0:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    test_acc_c100_d0 = 100 * correct / total
    print(f"\n🧪 Test Accuracy for c100_d0: {test_acc_c100_d0:.2f}%")

    return test_acc_c0_d100, test_acc_c100_d0

if __name__ == "__main__":
    run_tests()
