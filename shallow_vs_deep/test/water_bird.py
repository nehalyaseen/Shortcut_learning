import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np

def run_tests(model,test_loader_bg0b100, test_loader_bg100b0, device):
    # Testing
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader_bg0b100:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    test_acc_bg0b100 = 100 * correct / total
    print(f"\n🧪 Test Accuracy for bg0_b100: {test_acc_bg0b100:.2f}%")



    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader_bg100b0:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    test_acc_bg100b0 = 100 * correct / total
    print(f"\n🧪 Test Accuracy for bg100_b0: {test_acc_bg100b0:.2f}%")

    return test_acc_bg0b100, test_acc_bg100b0

if __name__ == "__main__":
    run_tests()
