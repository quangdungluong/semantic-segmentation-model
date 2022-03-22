import torch
from PIL import Image
import pandas as pd
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F


IMG_SIZE = 224
class SkinLesionDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_size=IMG_SIZE, mode='train', augmentation_prob=0.4):
        self.df = df
        self.image_size = IMG_SIZE
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.RotationDegree = [0, 90, 180, 270]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx, 0])
        mask = Image.open(self.df.iloc[idx, 1])
        
        aspect_ratio = image.size[1]/image.size[0]
        
        Transform = []
        p_transform = random.random()
        Transform.append(T.Resize((self.image_size, self.image_size)))
        
        if (self.mode=='train') and p_transform <= self.augmentation_prob:
            RotationDegree = self.RotationDegree[random.randint(0, 3)]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1/aspect_ratio
            Transform.append(T.RandomRotation(RotationDegree))
            
            Transform.append(T.RandomRotation(10))
            CropRange = random.randint(250,270)
            Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
            Transform = T.Compose(Transform)
            
            image = Transform(image)
            mask = Transform(mask)
            
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() < 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)
            
            Transform = []                
            image = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)(image)
            
        Transform.append(T.Resize((self.image_size, self.image_size)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        
        image = Transform(image)
        mask = Transform(mask)
        
        image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        return image, mask


class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_size = IMG_SIZE, mode = 'train', augmentation_prob = 0.4):
        self.df = df
        self.image_size = image_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.RotationDegree = [0, 90, 180, 270]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx, 0])
        mask = Image.open(self.df.iloc[idx, 1])
        
        aspect_ratio = image.size[1]/image.size[0]
        
        Transform = []
        p_transform = random.random()
        
        if self.mode=='train' and p_transform <= self.augmentation_prob:
            
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
                
            if random.random() < 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)
                
            image = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)(image)
        
        Transform = []
        Transform.append(T.Resize((self.image_size, self.image_size)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        
        image = Transform(image)
        mask = Transform(mask)
        
        image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        return image, mask