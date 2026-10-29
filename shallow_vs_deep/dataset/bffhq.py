import os, re
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataset(data_correlation, seed=42):
    set_seed(seed)
    female_young = "/home/ny1g23/denetdm/shallow_vs_deep/data/bffhq_split/female_young"
    female_old = "/home/ny1g23/denetdm/shallow_vs_deep/data/bffhq_split/female_old"
    male_old = "/home/ny1g23/denetdm/shallow_vs_deep/data/bffhq_split/male_old"
    male_young = "/home/ny1g23/denetdm/shallow_vs_deep/data/bffhq_split/male_young"

    meta_data="/home/ny1g23/denetdm/shallow_vs_deep/data/bffhq_split/gender_labels.csv"
    meta_df = pd.read_csv(meta_data)

    class bffhq_dataset(Dataset):
        def __init__(self, df, transform=None):
            self.df = df
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row["path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = row["gender"]
            return img, label
        
    def count_images(path, extensions={".jpg", ".jpeg", ".png"}):
        return sum(
            1 for f in os.listdir(path) 
            if os.path.splitext(f)[1].lower() in extensions
        )
    img_female_young=count_images(female_young)
    img_female_old=count_images(female_old)
    img_male_old=count_images(male_old)
    img_male_young=count_images(male_young)

    # helper to sample images from folder and assign label
    def sample_images(folder, n, label):
        all_imgs = os.listdir(folder)
        sampled = random.sample(all_imgs, n)
        df = meta_df[meta_df["image_number"].isin(sampled)].copy()
        df["gender"] = label
        df["path"] = df["image_number"].apply(lambda x: os.path.join(folder, x))
        return df
    print(data_correlation)

    if data_correlation in ["a30g100","a40g100", "a50g100", "a60g100","a70g100","a80g100","a90g100","a92g100","a94g100","a96g100","a98g100"]:
        gender_correlation, age_correlation = map(int, re.findall(r'\d+', data_correlation))
        print(gender_correlation, age_correlation)

        per_female_old = 1-(gender_correlation/100)
        per_female_young = 1-per_female_old
        print("per_female_young", per_female_young)
        print("per_female_old",per_female_old)

        base = min(img_female_young,img_female_old)
        
        
        num_female_young = int(base * per_female_young)
        num_female_old = int(base * per_female_old)
        print("num_female_young",num_female_young)
        print("num_female_old",num_female_old)

        print("------")

        per_male_young = 1 - (gender_correlation/100)    
        per_male_old = 1 - (per_male_young)

        print("per_male_old",per_male_old)
        print("per_male_young", per_male_young)
        base=min(img_male_young,img_male_old)
        num_male_old = int(base * per_male_old)
        num_male_young = int(base * per_male_young)

        df_female_old = sample_images(female_old, num_female_old, label=0)
        df_female_young = sample_images(female_young, num_female_young, label=0)
        df_male_old = sample_images(male_old, num_male_old, label=1)
        df_male_young =sample_images(male_young, num_male_young, label=1)

        dataset_df = pd.concat([df_female_old,df_female_young, df_male_old, df_male_young], ignore_index= True)
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        train_dataset = bffhq_dataset(dataset_df, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    if data_correlation in ["a100g100","a90g90","a92g92","a94g94","a96g96","a98g98"]:

        gender_correlation, age_correlation = map(int, re.findall(r'\d+', data_correlation))
        print(gender_correlation, age_correlation)

        samples=4000
        #num_female_young,num_female_old,num_male_young,num_male_old = int(samples//2),int(samples//2),int(samples//2),int(samples//2)

        num_female_young = int(samples*gender_correlation/100)
        num_female_old = int((1-(age_correlation/100))*samples)

        num_male_old = int(samples*gender_correlation/100)
        num_male_young = int((1-(age_correlation/100))*samples)

        df_female_old = sample_images(female_old, num_female_old, label=0)
        df_female_young = sample_images(female_young, num_female_young, label=0)
        df_male_old = sample_images(male_old, num_male_old, label=1)
        df_male_young =sample_images(male_young, num_male_young, label=1)

        dataset_df = pd.concat([df_female_old,df_female_young, df_male_old, df_male_young], ignore_index= True)
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        train_dataset = bffhq_dataset(dataset_df, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)   

    meta_test_a0g100= pd.read_csv("/home/ny1g23/denetdm/shallow_vs_deep/data/bffhq_split/a0g100/meta_data.csv")
    meta_test_a0g100["path"] = meta_test_a0g100["image_number"].apply(
        lambda x: os.path.join("/home/ny1g23/denetdm/shallow_vs_deep/data/bffhq_split/a0g100",x)
    )
    test_dataset_a0g100 = bffhq_dataset(meta_test_a0g100, transform=transform)
    test_loader_a0g100 = DataLoader(test_dataset_a0g100, batch_size=128, shuffle=True)

    meta_test_a100g0= pd.read_csv("/home/ny1g23/denetdm/shallow_vs_deep/data/bffhq_split/a100g0/meta_data.csv")
    meta_test_a100g0["path"] = meta_test_a100g0["image_number"].apply(
        lambda x: os.path.join("/home/ny1g23/denetdm/shallow_vs_deep/data/bffhq_split/a100g0",x)
    )
    test_dataset_a100g0 = bffhq_dataset(meta_test_a100g0, transform=transform)
    test_loader_a100g0 = DataLoader(test_dataset_a100g0, batch_size=128, shuffle=True)

    return train_loader, val_loader, test_loader_a0g100, test_loader_a100g0