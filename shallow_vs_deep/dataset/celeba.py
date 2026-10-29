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
    men_black = "/home/ny1g23/denetdm/shallow_vs_deep/data/celeba_split/men_black"
    men_blond = "/home/ny1g23/denetdm/shallow_vs_deep/data/celeba_split/men_blond"
    women_black = "/home/ny1g23/denetdm/shallow_vs_deep/data/celeba_split/women_black"
    women_blond = "/home/ny1g23/denetdm/shallow_vs_deep/data/celeba_split/women_blond"
    meta_data="/home/ny1g23/denetdm/shallow_vs_deep/data/celeba_split/list_attr_celeba.csv"
    meta_df = pd.read_csv(meta_data)

    class celeba_dataset(Dataset):
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
    img_men_black=count_images(men_black)
    img_men_blond=count_images(men_blond)
    img_women_black=count_images(women_black)
    img_women_blond=count_images(women_blond)

    def invert_labels(df, n_invert):
        if n_invert > 0:
            idx_to_flip = df.sample(n=n_invert, random_state=42).index
            df.loc[idx_to_flip, "gender"] = 1 - df.loc[idx_to_flip, "gender"]
        return df
    
    # helper to sample images from folder and assign label
    def sample_images(folder, n, label):
        all_imgs = os.listdir(folder)
        sampled = random.sample(all_imgs, n)
        df = meta_df[meta_df["image_id"].isin(sampled)].copy()
        df["gender"] = label
        df["path"] = df["image_id"].apply(lambda x: os.path.join(folder, x))
        return df


    if data_correlation in ["h30g100","h40g100","h50g100", "h60g100", "h70g100", "h80g100", "h90g100", "h92g100", "h94g100", "h96g100", "h98g100", "h100g100"]:
        hair_correlation, gender_correlation = map(int, re.findall(r'\d+', data_correlation))
        print(hair_correlation, gender_correlation)

        per_men_blond = 1 - (hair_correlation/100)
        per_men_black = 1 - per_men_blond
        #print("per_men_blond", per_men_blond)
        #print("per_men_black", per_men_black)

        base = min(img_men_blond, img_men_black)
        #print("base:", base)

        num_men_blond = int(base* per_men_blond)
        num_men_black = int(base* per_men_black)
        #print("num_men_blond:", num_men_blond)
        #print("num_men_black:", num_men_black)

        per_women_black = 1- (hair_correlation/100)
        per_women_blond = 1- per_women_black
        #print("per_women_black:",per_women_black)
        #print("per_women_blond",per_women_blond)

        base= min(img_women_black, img_women_blond)
        #print("base:", base)

        num_women_black = int(base * per_women_black)
        num_women_blond = int(base * per_women_blond)
        #print("num_women_black",num_women_black)
        #print("num_women_blond",num_women_blond)

        df_men_black = sample_images(men_black, num_men_black, label=1)
        df_men_blond = sample_images(men_blond, num_men_blond, label=1)
        df_women_blond = sample_images(women_blond, num_women_blond, label=0)
        df_women_black = sample_images(women_black, num_women_black, label=0)

        dataset_df = pd.concat([df_men_black, df_men_blond, df_women_black, df_women_blond], ignore_index= True)
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        train_dataset = celeba_dataset(dataset_df, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    if data_correlation in ["h90g90","h92g92", "h94g94", "h96g96", "h98g98"]:
        hair_correlation, gender_correlation = map(int, re.findall(r'\d+', data_correlation))
        #print(hair_correlation, gender_correlation)

        samples=1574
        num_men_blond, num_men_black, num_women_blond, num_women_black= int(samples//2),int(samples//2),int(samples//2),int(samples//2)
        #print("training data:", int(samples//2)+int(samples//2)+int(samples//2)+int(samples//2))
        #print(num_men_blond, num_men_black, num_women_blond, num_women_black)
        men_inverted_labels=int((1-(gender_correlation/100))*samples)
        women_inverted_labels=int((1-(gender_correlation/100))*samples)

        df_men_black = sample_images(men_black, num_men_black, label=1)
        df_men_blond = sample_images(men_blond, num_men_blond, label=1)
        df_women_blond = sample_images(women_blond, num_women_blond, label=0)
        df_women_black = sample_images(women_black, num_women_black, label=0)

        df_all = pd.concat([df_men_black, df_men_blond, df_women_black, df_women_blond], ignore_index= True)

        df_men = df_all[df_all["gender"] == 1]
        df_women = df_all[df_all["gender"] == 0]

        df_men = invert_labels(df_men, men_inverted_labels)
        df_women =  invert_labels(df_women, women_inverted_labels)
        
        dataset_df = pd.concat([df_men, df_women], ignore_index=True)
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        train_dataset = celeba_dataset(dataset_df, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)        

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    meta_test_h0g100= pd.read_csv("/home/ny1g23/denetdm/shallow_vs_deep/data/celeba_split/h0g100/meta_data.csv")
    meta_test_h0g100["path"] = meta_test_h0g100["image_id"].apply(
        lambda x: os.path.join("/home/ny1g23/denetdm/shallow_vs_deep/data/celeba_split/h0g100",x)
    )
    test_dataset_h0g100 = celeba_dataset(meta_test_h0g100, transform=transform)
    test_loader_h0g100 = DataLoader(test_dataset_h0g100, batch_size=128, shuffle=True)

    meta_test_h100g0= pd.read_csv("/home/ny1g23/denetdm/shallow_vs_deep/data/celeba_split/h100g0/meta_data.csv")
    meta_test_h100g0["path"] = meta_test_h100g0["image_id"].apply(
        lambda x: os.path.join("/home/ny1g23/denetdm/shallow_vs_deep/data/celeba_split/h100g0",x)
    )
    test_dataset_h100g0 = celeba_dataset(meta_test_h100g0, transform=transform)
    test_loader_h100g0 = DataLoader(test_dataset_h100g0, batch_size=128, shuffle=True)

    return train_loader, val_loader, test_loader_h0g100, test_loader_h100g0