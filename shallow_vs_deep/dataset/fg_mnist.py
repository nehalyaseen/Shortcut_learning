import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import random
import os
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataset(train_color_correlation, train_digit_correlation, seed=42, batch_size=128):

    set_seed(seed)    
    #train_color_correlation = int(train_color_correlation.split('c')[1]) #30
    #train_digit_correlation = int(train_digit_correlation.split('d')[1]) #d100  


    DIGIT_COLOUR_MAP = {
        0: (255, 0, 0),     # Red
        1: (0, 255, 0),     # Green
        2: (0, 0, 255),     # Blue
        3: (255, 255, 0),   # Yellow
        4: (255, 0, 255),   # Magenta
        5: (0, 255, 255),   # Cyan
        6: (128, 0, 128),   # Purple
        7: (255, 165, 0),   # Orange
        8: (128, 128, 128), # Grey
        9: (0, 128, 128),   # Teal
    }

    def colour_background_black_digit(image, bg_colour, variation=30):
        image = image.convert('RGB')
        pixels = image.load()

        r_base, g_base, b_base = bg_colour
        r_var = np.clip(r_base + np.random.randint(-variation, variation + 1), 0, 255)
        g_var = np.clip(g_base + np.random.randint(-variation, variation + 1), 0, 255)
        b_var = np.clip(b_base + np.random.randint(-variation, variation + 1), 0, 255)

        for i in range(image.size[0]):
            for j in range(image.size[1]):
                gray = pixels[i, j][0]
                if gray < 10:
                    pixels[i, j] = (0,0,0)
                else:
                    pixels[i, j] = (r_var, g_var, b_var)

        return image

    class DigitColourCorrelatedMNIST(Dataset):
        def __init__(self, root, train=True, transform=None, download=True, 
                    digit_correlation=1.0, colour_correlation=1.0):
            self.dataset = MNIST(root=root, train=train, download=download)
            self.transform = transform
            self.digit_corr = digit_correlation
            self.colour_corr = colour_correlation

            self.label_to_imgs = {i: [] for i in range(10)}
            for img, label in self.dataset:
                self.label_to_imgs[label].append(img)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            _, label = self.dataset[idx]

            # Select digit shape
            if random.random() < self.digit_corr:
                digit_label = label
            else:
                digit_label = random.choice([i for i in range(10) if i != label])

            digit_image = random.choice(self.label_to_imgs[digit_label])

            # Select digit colour
            if random.random() < self.colour_corr:
                colour_label = label
            else:
                colour_label = random.choice([i for i in range(10) if i != label])

            digit_colour = DIGIT_COLOUR_MAP[colour_label]
            coloured_image = colour_background_black_digit(digit_image, digit_colour)

            if self.transform:
                coloured_image = self.transform(coloured_image)

            return coloured_image, label



    #for training data
    full_train_dataset = DigitColourCorrelatedMNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        colour_correlation=train_color_correlation/100,
        digit_correlation=train_digit_correlation/100       
            
    )

    test_c0d100 = DigitColourCorrelatedMNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor(),
        colour_correlation=0,
        digit_correlation=1  
    )

    test_c100d0 = DigitColourCorrelatedMNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor(),
        colour_correlation=1,
        digit_correlation=0  
    )


    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    seed=42
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


    batch_size=128
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_c0d100 = DataLoader(test_c0d100, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_c100d0 = DataLoader(test_c100d0, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader_c0d100, test_loader_c100d0