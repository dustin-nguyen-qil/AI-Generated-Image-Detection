import torch
from torch import nn 
from torch.utils.data import Dataset
from PIL import Image
import os
import os.path as osp

from utils import get_rich_and_poor_images

class ImageDataset(Dataset):
    def __init__(self, img_transform, data_dir, patch_size) -> None:
        super().__init__()
        self.img_transform = img_transform
        self.data_dir = data_dir
        self.patch_size = patch_size

        # get image samples
        self.data = self.get_data()
    
    def get_data(self):
        """
            Get images from data folder 
        """
        data = []
        imgs = [file for file in os.listdir(self.data_dir) if file.lower().endswith('.jpeg')]
        for img in imgs:
            img_path = osp.join(self.data_dir, img)
            if img_path.startswith('dm') or img_path.startswith('gan'):
                label = 0 # fake 
            else:
                label = 1 # real
            data.append((img_path, label))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path)
        img_transformed = self.img_transform(img)

        # get rich texture image and poor texture image from the original image
        rich_image, poor_image = get_rich_and_poor_images(img_transformed, self.patch_size)
        return rich_image, poor_image, label 