import os
import numpy as np

import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch

import glob
import random
from torch.utils.data import Dataset
from PIL import Image

def convert_to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        # Load file lists
        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B") + "/*.*"))

        # Check if directories contain images
        if not self.files_A:
            raise ValueError(f"No images found in {os.path.join(root, f'{mode}A')}. Please check the dataset directory.")
        if not self.files_B:
            raise ValueError(f"No images found in {os.path.join(root, f'{mode}B')}. Please check the dataset directory.")

    def __getitem__(self, index):
        # Load image from domain A
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        # Load image from domain B (unaligned or aligned)
        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert to RGB if needed
        if image_A.mode != "RGB":
            image_A = convert_to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = convert_to_rgb(image_B)

        # Apply transformations
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {"A": item_A, "B": item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
class ReplayBuffer:
    def __init__(self, max_size = 50):
        assert max_size > 0, "Empty buffer"
        self.max_size = max_size
        self.data = []

    def push_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)                
            else:
                # returns a newly added image with a probability of 0.5
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                # returns an older generated image
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert(
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        # Checks whether the current epoch has exceeded the decay epoch
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
        

def initialize_weights(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)