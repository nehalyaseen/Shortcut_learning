import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import random
from PIL import Image
import os
import numpy as np
from config import transform_domino

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataset(train_mnist_correlation, train_cifar_correlation, model_tag,seed=42, batch_size=128):

    set_seed(seed)


    class MNISTCIFARCollage(Dataset):
        def __init__(self, mnist_root, cifar_root, save_dir=None, train=True, mnist_corr=1.0, cifar_corr=1.0, transform=None):
            """
            mnist_corr: Probability MNIST digit matches the final label.
            cifar_corr: Probability CIFAR image matches the final label.
            save_dir: If given, generated collages will be saved here as PNG + a labels.txt file.
            """
            self.mnist = datasets.MNIST(mnist_root, train=train, download=True)
            self.cifar = datasets.CIFAR10(cifar_root, train=train, download=True)
            self.mnist_corr = mnist_corr
            self.cifar_corr = cifar_corr
            self.transform = transform
            self.save_dir = save_dir
            
            # Group indices by class for sampling
            self.mnist_class_indices = {i: [] for i in range(10)}
            for idx, (_, label) in enumerate(self.mnist):
                self.mnist_class_indices[label].append(idx)
            
            self.cifar_class_indices = {i: [] for i in range(10)}
            for idx, (_, label) in enumerate(self.cifar):
                self.cifar_class_indices[label].append(idx)

            # Prepare save directory
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                self.labels_file = open(os.path.join(self.save_dir, "labels.txt"), "w")
        
        def __len__(self):
            return min(len(self.mnist), len(self.cifar))
        
        def _sample_from_class(self, dataset, class_indices, target_class):
            idx = random.choice(class_indices[target_class])
            return dataset[idx][0]

        def __getitem__(self, idx):
            # Decide final label for this composite sample
            label = random.randint(0, 9)
            
            # Sample MNIST
            if random.random() < self.mnist_corr:
                mnist_img = self._sample_from_class(self.mnist, self.mnist_class_indices, label)
            else:
                other_label = random.choice([i for i in range(10) if i != label])
                mnist_img = self._sample_from_class(self.mnist, self.mnist_class_indices, other_label)
            
            # Sample CIFAR
            if random.random() < self.cifar_corr:
                cifar_img = self._sample_from_class(self.cifar, self.cifar_class_indices, label)
            else:
                other_label = random.choice([i for i in range(10) if i != label])
                cifar_img = self._sample_from_class(self.cifar, self.cifar_class_indices, other_label)
            
            # Resize MNIST to match CIFAR width
            mnist_img = mnist_img.resize((32, 32), Image.BILINEAR).convert('RGB')
            
            # Combine vertically
            collage = Image.new('RGB', (32, 64))
            #collage.paste(mnist_img, (0, 0))
            #collage.paste(cifar_img, (0, 32))
            
            collage.paste(cifar_img, (0, 0))
            collage.paste(mnist_img, (0, 32))

            if self.transform:
                collage = self.transform(collage)
            
            return collage, label

        def close(self):
            if self.save_dir:
                self.labels_file.close()


    

    full_train_dataset = MNISTCIFARCollage(
        mnist_root="./data",
        cifar_root="./data",
        train=True,
        mnist_corr=train_mnist_correlation/100,
        cifar_corr=train_cifar_correlation/100,
        transform=None
    )    

    test_m0c100 = MNISTCIFARCollage(
        mnist_root="./data",
        cifar_root="./data",
        train=False,
        mnist_corr=0,
        cifar_corr=1,
        transform=None
    )    

    test_m100c0 = MNISTCIFARCollage(
        mnist_root="./data",
        cifar_root="./data",
        train=False,
        mnist_corr=1,
        cifar_corr=0,
        transform=None
    )    

    if model_tag == "vit":
        transform = transforms.Compose([
        transforms.Resize((64, 32)),    # resize to 64x32
        transforms.Pad((16, 0, 16, 0)), # pad to 64x64
        transforms.ToTensor()
        ])
        full_train_dataset = transform_domino(full_train_dataset, transform)
        test_m0c100 = transform_domino(test_m0c100, transform)
        test_m100c0 = transform_domino(test_m100c0, transform)

        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        seed=42
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
        
        batch_size = 128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader_m0c100 = DataLoader(test_m0c100, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader_m100c0 = DataLoader(test_m100c0, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if model_tag in ["mlp3", "mlp5", "resnet" ,"vgg" ]:

        transform = transforms.Compose([
        transforms.ToTensor()
        ])
        full_train_dataset = transform_domino(full_train_dataset, transform)
        test_m0c100 = transform_domino(test_m0c100, transform)
        test_m100c0 = transform_domino(test_m100c0, transform)

        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        seed=42
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
        
        batch_size = 128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader_m0c100 = DataLoader(test_m0c100, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader_m100c0 = DataLoader(test_m100c0, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader_m0c100, test_loader_m100c0