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
    waterbird_on_water = "/home/ny1g23/denetdm/shallow_vs_deep/data/waterbird_split/waterbird on water"
    waterbird_on_land = "/home/ny1g23/denetdm/shallow_vs_deep/data/waterbird_split/waterbird on land"
    landbird_on_water = "/home/ny1g23/denetdm/shallow_vs_deep/data/waterbird_split/landbird on water"
    landbird_on_land = "/home/ny1g23/denetdm/shallow_vs_deep/data/waterbird_split/landbird on land"
    meta_data = "/home/ny1g23/denetdm/shallow_vs_deep/data/waterbird_split/my_metadata.csv"
    meta_df = pd.read_csv(meta_data)
    # custom dataset class
    class BirdDataset(Dataset):
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
            label = row["label"]
            return img, label

    def count_images(path, extensions={".jpg", ".jpeg", ".png"}):
        return sum(
        1 for f in os.listdir(path) 
        if os.path.splitext(f)[1].lower() in extensions
        )
    img_waterbird_on_land=count_images(waterbird_on_land)
    img_landbird_on_water=count_images(landbird_on_water)
    img_waterbird_on_water=count_images(waterbird_on_water)
    img_landbird_on_land=count_images(landbird_on_land)

    def invert_labels(df, n_invert):
        if n_invert > 0:
            idx_to_flip = df.sample(n=n_invert, random_state=42).index
            df.loc[idx_to_flip, "label"] = 1 - df.loc[idx_to_flip, "label"]
        return df
    
    # helper to sample images from folder and assign label
    def sample_images(folder, n, label):
        all_imgs = os.listdir(folder)
        sampled = random.sample(all_imgs, n)
        df = meta_df[meta_df["img_filename"].isin(sampled)].copy()
        df["label"] = label
        df["path"] = df["img_filename"].apply(lambda x: os.path.join(folder, x))
        return df


    if data_correlation in ["bg30b100", "bg40b100", "bg50b100", "bg60b100", "bg70b100", "bg80b100", "bg90b100", "bg92b100", "bg94b100", "bg96b100", "bg98b100", "bg100b100"]:
        bg_correlation, bird_correlation = map(int, re.findall(r'\d+', data_correlation))

        # proportions always sum to 1
        per_waterbird_on_land = 1 - (bg_correlation / 100)
        per_waterbird_on_water = 1 - per_waterbird_on_land

        # common base ensures equal total counts
        base = 748

        # apply proportions
        num_waterbird_on_water = int(base * per_waterbird_on_water)
        num_waterbird_on_land = int(base * per_waterbird_on_land)


        per_landbird_on_water = 1 - (bg_correlation / 100)
        per_landbird_on_land = 1 - per_landbird_on_water

        # apply proportions
        num_landbird_on_water = int(base * per_landbird_on_water)
        num_landbird_on_land = int(base * per_landbird_on_land)


        # build dataset dataframe
        df_water_on_water = sample_images(waterbird_on_water, num_waterbird_on_water, label=1)
        df_water_on_land  = sample_images(waterbird_on_land,  num_waterbird_on_land,  label=1)
        df_land_on_water  = sample_images(landbird_on_water,  num_landbird_on_water,  label=0)
        df_land_on_land   = sample_images(landbird_on_land,   num_landbird_on_land,   label=0)

        dataset_df = pd.concat([df_water_on_water, df_water_on_land,
                                df_land_on_water, df_land_on_land], ignore_index=True)

        # usage
        # from torchvision import transforms
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

        train_dataset = BirdDataset(dataset_df, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        images, labels = next(iter(train_loader))

    if data_correlation in ["bg90b90", "bg92b92", "bg94b94", "bg96b96", "bg98b98"]:
        bg_correlation, bird_correlation = map(int, re.findall(r'\d+', data_correlation))
        
        total=748

        total_waterbird_on_land, total_waterbird_on_water, total_landbird_on_land ,total_landbird_on_water= total// 2, total// 2, total// 2, total// 2


        # number of labels to invert
        waterbird_inverted_labels = int((1 - (bird_correlation / 100)) * total)
        landbird_inverted_labels  = int((1 - (bird_correlation / 100)) * total)


        # sample dataframes with original labels
        df_water_on_water = sample_images(waterbird_on_water, total_waterbird_on_water, label=1)
        df_water_on_land  = sample_images(waterbird_on_land,  total_waterbird_on_land,  label=1)
        df_land_on_water  = sample_images(landbird_on_water,  total_landbird_on_water,  label=0)
        df_land_on_land   = sample_images(landbird_on_land,   total_landbird_on_land,   label=0)

        df_all = pd.concat([
            df_water_on_water,
            df_water_on_land,
            df_land_on_water,
            df_land_on_land
        ], ignore_index=True)

        # split waterbirds and landbirds
        df_waterbirds = df_all[df_all["label"] == 1]
        df_landbirds  = df_all[df_all["label"] == 0]

        # invert labels
        df_waterbirds = invert_labels(df_waterbirds, waterbird_inverted_labels)
        df_landbirds  = invert_labels(df_landbirds, landbird_inverted_labels)

        # combine back
        dataset_df = pd.concat([df_waterbirds, df_landbirds], ignore_index=True)

        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

        train_dataset = BirdDataset(dataset_df, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- load test_bg0b100
    meta_test_bg0b100 = pd.read_csv("/home/ny1g23/denetdm/shallow_vs_deep/data/waterbird_split/test_bg0b100/metadata.csv")
    meta_test_bg0b100["path"] = meta_test_bg0b100["img_filename"].apply(
        lambda x: os.path.join("/home/ny1g23/denetdm/shallow_vs_deep/data/waterbird_split/test_bg0b100", x)
    )
    test_dataset_bg0b100 = BirdDataset(meta_test_bg0b100, transform=transform)
    test_loader_bg0b100 = DataLoader(test_dataset_bg0b100, batch_size=128, shuffle=True)

    # --- load test_bg100b0
    meta_test_bg100b0 = pd.read_csv("/home/ny1g23/denetdm/shallow_vs_deep/data/waterbird_split/test_bg100b0/metadata.csv")
    meta_test_bg100b0["path"] = meta_test_bg100b0["img_filename"].apply(
        lambda x: os.path.join("/home/ny1g23/denetdm/shallow_vs_deep/data/waterbird_split/test_bg100b0", x)
    )
    test_dataset_bg100b0 = BirdDataset(meta_test_bg100b0, transform=transform)
    test_loader_bg100b0 = DataLoader(test_dataset_bg100b0, batch_size=128, shuffle=True)

    return train_loader, val_loader, test_loader_bg0b100, test_loader_bg100b0




