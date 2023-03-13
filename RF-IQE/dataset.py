import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MyDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        super(MyDataset, self).__init__()
        self.transform = transform
        A_images = os.listdir(root_A)
        B_images = os.listdir(root_B)
        
        self.A_images = [os.path.join(root_A, img) for img in A_images]
        self.B_images = [os.path.join(root_B, img) for img in B_images]
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)
        self.length_dataset = max(self.A_len, self.B_len)

    def __len__(self):
        return self.length_dataset
        
    def add_elements_to_A(self, new_root_A):
        new_A_images = os.listdir(new_root_A)
        
        for img in new_A_images:
          self.A_images.append(os.path.join(new_root_A,img))
        self.A_len = len(self.A_images)
        self.length_dataset = max(self.A_len, self.B_len)
        
    def add_elements_to_B(self, new_root_B):
        new_B_images = os.listdir(new_root_B)
        
        for img in new_B_images:
          self.B_images.append(os.path.join(new_root_B,img))
        self.B_len = len(self.B_images)
        self.length_dataset = max(self.A_len, self.B_len)
        
    
    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        B_img = self.B_images[index % self.B_len]

        A_img = np.array(Image.open(A_img).convert("RGB"))
        B_img = np.array(Image.open(B_img).convert("RGB"))

        if self.transform:
            augmentataions = self.transform(image = A_img, image0=B_img)
            A_img = augmentataions["image"]
            B_img = augmentataions["image0"]

        return A_img, B_img


class FundusDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        super(FundusDataset, self).__init__()
        self.transform = transform
        A_images = os.listdir(root_A)
        B_images = os.listdir(root_B)
        
        self.A_images = [os.path.join(root_A, img) for img in A_images]
        self.B_images = [os.path.join(root_B, img) for img in B_images]
        self.A_label = [int(root_A[-1]) for _ in self.A_images]
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)
        self.length_dataset = max(self.A_len, self.B_len)

    def __len__(self):
        return self.length_dataset
        
    def add_elements_to_A(self, new_root_A):
        new_A_images = os.listdir(new_root_A)
        
        for img in new_A_images:
          self.A_images.append(os.path.join(new_root_A,img))
          self.A_label.append(int(new_root_A[-1]))
        self.A_len = len(self.A_images)
        self.length_dataset = max(self.A_len, self.B_len)
        
    def add_elements_to_B(self, new_root_B):
        new_B_images = os.listdir(new_root_B)
        
        for img in new_B_images:
          self.B_images.append(os.path.join(new_root_B,img))
        self.B_len = len(self.B_images)
        self.length_dataset = max(self.A_len, self.B_len)
        
    
    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        A_label = self.A_label[index % self.A_len]
        B_img = self.B_images[index % self.B_len]

        A_img = np.array(Image.open(A_img).convert("RGB"))
        B_img = np.array(Image.open(B_img).convert("RGB"))

        if self.transform:
            augmentataions = self.transform(image = A_img, image0=B_img)
            A_img = augmentataions["image"]
            B_img = augmentataions["image0"]

        return A_img, B_img, A_label

