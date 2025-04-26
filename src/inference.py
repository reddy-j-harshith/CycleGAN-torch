import sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

from utils import *
from CycleGAN import *

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hyper = Hyperparameters(
    img_size=128,
    channels=3,
    num_resi_blocks=9,
)

# Load models
input_shape = (hyper.channels, hyper.img_size, hyper.img_size)

Gen_AB = Generator(input_shape=input_shape, num_resi_blocks=hyper.num_resi_blocks).to(device)
Gen_BA = Generator(input_shape=input_shape, num_resi_blocks=hyper.num_resi_blocks).to(device)

# Load checkpoint
if len(sys.argv) < 3:
    print("Usage: python inference.py <checkpoint_epoch> <image_path> <AtoB or BtoA>")
    sys.exit(1)

start_epoch = int(sys.argv[1])
image_path = sys.argv[2]
direction = sys.argv[3]  # either "AtoB" or "BtoA"

checkpoint_dir = f"../checkpoints/epoch{start_epoch}"
Gen_AB.load_state_dict(torch.load(f"{checkpoint_dir}/Gen_AB_epoch_{start_epoch}.pth", map_location=device))
Gen_BA.load_state_dict(torch.load(f"{checkpoint_dir}/Gen_BA_epoch_{start_epoch}.pth", map_location=device))

Gen_AB.eval()
Gen_BA.eval()

# Transformation
transform = transforms.Compose([
    transforms.Resize((hyper.img_size, hyper.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load and transform the image
image = Image.open(image_path).convert("RGB")
input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Perform inference
with torch.no_grad():
    if direction == "AtoB":
        output_image = Gen_AB(input_image)
    elif direction == "BtoA":
        output_image = Gen_BA(input_image)
    else:
        raise ValueError("Direction must be either 'AtoB' or 'BtoA'")

# Unnormalize and display the output
output_image = output_image.squeeze(0).cpu()
output_image = output_image * 0.5 + 0.5  # Denormalize to [0,1]

output_image = transforms.ToPILImage()(output_image)

plt.figure(figsize=(6,6))
plt.imshow(output_image)
plt.axis('off')
plt.show()
