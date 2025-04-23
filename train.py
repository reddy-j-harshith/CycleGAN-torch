import numpy as np
import itertools
import time
import datetime
import tqdm
import json
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mping

from utils import *
from CycleGAN import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hyper = Hyperparameters(
    epoch = 1,
    n_epochs = 200,
    dataset_train_mode = "train",
    dataset_test_mode = "test",
    batch_size = 4,
    lr = 0.0002,
    decay_start_epoch = 100,
    b1 = 0.5,
    b2 = 0.999,
    n_cpu = 8,
    img_size = 128,
    channels = 3,
    sample_interval = 100,
    num_resi_blocks = 9,
    lambda_cyc = 15.0,
    lambda_id = 10.0
)

# file_path = "./datasets/summer2winter"

file_path = "./datasets/summer2winter"

print(file_path)

def show_img(img, size = 10):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize = (size, size))

    # Pytorch image processing module expects the tensors to be in Channel x Height x Width format.
    # Whereas PIL and matplotlib expects it to be in H x W x C
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def to_img(x):
    x = x.view(x.size(0) * 2, hyper.channels, hyper.img_size, hyper.img_size)
    return x

def plot_output(path, x, y):
    img = mping.imread(path)
    plt.figure(figsize=(x, y))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

transforms_ = [
    transforms.Resize((hyper.img_size, hyper.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_loader = DataLoader(
    ImageDataset(root = file_path, mode = hyper.dataset_train_mode, transforms_ = transforms_, unaligned = True),
    batch_size = hyper.batch_size,
    shuffle = True,
    num_workers = 2,
)

val_loader = DataLoader(
    ImageDataset(root = file_path, mode = hyper.dataset_test_mode, transforms_ = transforms_, unaligned = True),
    batch_size = 16,
    shuffle = True,
    num_workers = 2,
)

def save_img_samples(batches_done):
    print("batches_done ", batches_done)
    imgs = next(iter(val_loader))

    Gen_AB.eval()
    Gen_BA.eval()

    real_A = imgs["A"].to(device)
    fake_B = Gen_AB(real_A)

    real_B = imgs["B"].to(device)
    fake_A = Gen_BA(real_B)

    real_A = make_grid(real_A, nrow=16, normalize=True)
    real_B = make_grid(real_B, nrow=16, normalize=True)
    fake_A = make_grid(fake_A, nrow=16, normalize=True)
    fake_B = make_grid(fake_B, nrow=16, normalize=True)

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    output_dir = "./output_images"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{batches_done}.png")

    save_image(image_grid, path, normalize=False)
    return path

# normal GAN Loss
GAN_loss = torch.nn.MSELoss()

# Cycle Consistency Loss
cycle_loss = torch.nn.L1Loss()

# Cycle Identity Loss
identity_loss = torch.nn.L1Loss()

input_shape = (hyper.channels, hyper.img_size, hyper.img_size)

Gen_AB = Generator(input_shape = input_shape, num_resi_blocks = hyper.num_resi_blocks)
Gen_BA = Generator(input_shape = input_shape, num_resi_blocks = hyper.num_resi_blocks)
Disc_A = Discriminator(input_shape = input_shape)
Disc_B = Discriminator(input_shape = input_shape)

if torch.cuda.is_available():
    Gen_AB = Gen_AB.to(device)
    Gen_BA = Gen_BA.to(device)
    Disc_A = Disc_A.to(device)
    Disc_B = Disc_B.to(device)
    GAN_loss.to(device)
    cycle_loss.to(device)
    identity_loss.to(device)

# INITIALIZE WEIGHTS

Gen_AB.apply(initialize_weights)
Gen_BA.apply(initialize_weights)
Disc_A.apply(initialize_weights)
Disc_B.apply(initialize_weights)

# BUFFER CREATION

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Defining the optimizers

optimizer_G = torch.optim.Adam(
    # Passing the combined parameters of both the Generators
    itertools.chain(Gen_AB.parameters(), Gen_BA.parameters()),
    lr = hyper.lr,
    betas = (hyper.b1, hyper.b2)
)

optimizer_Disc_A = torch.optim.Adam(
    Disc_A.parameters(), 
    lr = hyper.lr, 
    betas = (hyper.b1, hyper.b2)
)

optimizer_Disc_B = torch.optim.Adam(
    Disc_B.parameters(),
    lr = hyper.lr,
    betas = (hyper.b1, hyper.b2)
)

# LEARNING RATE SCHEDULERS

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer = optimizer_G,
    lr_lambda = LambdaLR(hyper.n_epochs, hyper.epoch, hyper.decay_start_epoch).step,
)

lr_scheduler_Disc_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer = optimizer_Disc_A,
    lr_lambda = LambdaLR(hyper.n_epochs, hyper.epoch, hyper.decay_start_epoch).step,
)

lr_scheduler_Disc_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer = optimizer_Disc_B,
    lr_lambda = LambdaLR(hyper.n_epochs, hyper.epoch, hyper.decay_start_epoch).step,
)

# Function to load existing JSON logs or initialize a new list
def load_json_logs(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return []

# Function to save logs to JSON
def save_json_logs(json_path, logs):
    with open(json_path, 'w') as f:
        json.dump(logs, f, indent=4)

# FINAL TRAINING FUNCTION

def train(
    Gen_AB: Generator,
    Gen_BA: Generator,
    Disc_A: Discriminator,
    Disc_B: Discriminator,
    train_dataloader: DataLoader,
    n_epochs: int,
    identity_loss: torch.nn.L1Loss,
    cycle_loss: torch.nn.L1Loss,
    GAN_loss: torch.nn.MSELoss,
    lambda_cyc: float,
    lambda_id: float,
    fake_A_Buffer: ReplayBuffer,
    fake_B_Buffer: ReplayBuffer,
    optimizer_G: torch.optim.Adam,
    optimizer_Disc_A: torch.optim.Adam,
    optimizer_Disc_B: torch.optim.Adam,
    sample_interval: int,
):
    prev_time = time.time()
    json_path = "./checkpoints/training_logs.json"
    training_logs = load_json_logs(json_path)  # Load existing logs or initialize empty list

    for epoch in range(hyper.epoch, n_epochs):
        for i, batch in enumerate(train_dataloader):

            # Set model input
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Adversarial Ground Truths
            valid = torch.ones((real_A.size(0), *Disc_A.output_shape), device=device, requires_grad=False)
            fake = torch.zeros((real_A.size(0), *Disc_A.output_shape), device=device, requires_grad=False)


            # TRAIN GENERATORS


            Gen_AB.train()
            Gen_BA.train()

            # Make the gradients to zero before the next step
            optimizer_G.zero_grad()

            # Pass the real image, to learn self-domain.
            # Basically, it should not disturb the contents 
            # which are already in the same domain
            loss_id_A = identity_loss(Gen_BA(real_A), real_A)
            loss_id_B = identity_loss(Gen_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # Identify fake images using the Discriminators
            fake_B = Gen_AB(real_A)
            loss_GAN_AB = GAN_loss(Disc_B(fake_B), valid)

            fake_A = Gen_BA(real_B)
            loss_GAN_BA = GAN_loss(Disc_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle consistency loss
            reconstructed_A = Gen_BA(fake_B)
            loss_cycle_A = cycle_loss(reconstructed_A, real_A)

            reconstructed_B = Gen_AB(fake_A)
            loss_cycle_B = cycle_loss(reconstructed_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            loss_G.backward()

            optimizer_G.step()


            # TRAIN DISCRIMINATOR A


            optimizer_Disc_A.zero_grad()

            loss_real = GAN_loss(Disc_A(real_A), valid)
            fake_A_ = fake_A_Buffer.push_pop(fake_A)

            loss_fake = GAN_loss(Disc_A(fake_A_.detach()), fake)
            loss_Disc_A = (loss_real + loss_fake) / 2

            loss_Disc_A.backward()
            optimizer_Disc_A.step()


            # TRAIN DISCRIMINATOR B


            optimizer_Disc_B.zero_grad()

            loss_real = GAN_loss(Disc_B(real_B), valid)
            fake_B_ = fake_B_Buffer.push_pop(fake_B)

            loss_fake = GAN_loss(Disc_B(fake_B_.detach()), fake)
            loss_Disc_B = (loss_real + loss_fake) / 2

            loss_Disc_B.backward()
            optimizer_Disc_B.step()

            loss_D = (loss_Disc_A + loss_Disc_B) / 2

            # Log metrics
            batches_done = epoch * len(train_dataloader) + i
            batches_left = n_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

            log_entry = {
                "epoch": epoch,
                "batch": i,
                "batches_done": batches_done,
                "D_loss": loss_D.item(),
                "G_loss": loss_G.item(),
                "GAN_loss": loss_GAN.item(),
                "cycle_loss": loss_cycle.item(),
                "identity_loss": loss_identity.item(),
                "ETA": str(time_left)
            }
            training_logs.append(log_entry)

            # Save logs periodically (e.g., every 100 batches)
            if batches_done % 100 == 0:
                save_json_logs(json_path, training_logs)

            prev_time = time.time()

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )


            # If at sample interval save image
            if batches_done % sample_interval == 0:
                save_img_samples(batches_done)
   
        checkpoint_dir = f"./checkpoints/epoch{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(Gen_AB.state_dict(), f"{checkpoint_dir}/Gen_AB_epoch_{epoch}.pth")
        torch.save(Gen_BA.state_dict(), f"{checkpoint_dir}/Gen_BA_epoch_{epoch}.pth")
        torch.save(Disc_A.state_dict(), f"{checkpoint_dir}/Disc_A_epoch_{epoch}.pth")
        torch.save(Disc_B.state_dict(), f"{checkpoint_dir}/Disc_B_epoch_{epoch}.pth")

        lr_scheduler_G.step()
        lr_scheduler_Disc_A.step()
        lr_scheduler_Disc_B.step()

        # Save logs at the end of each epoch
        save_json_logs(json_path, training_logs)

if __name__ == '__main__':
    train(
        Gen_AB =            Gen_AB,
        Gen_BA =            Gen_BA,
        Disc_A =            Disc_A,
        Disc_B =            Disc_B,
        train_dataloader =  train_loader,
        n_epochs =          hyper.n_epochs,
        identity_loss =     identity_loss,
        cycle_loss =        cycle_loss,
        GAN_loss =          GAN_loss,
        lambda_cyc=hyper.lambda_cyc,
        lambda_id=hyper.lambda_id,
        fake_A_Buffer=fake_A_buffer,
        fake_B_Buffer=fake_B_buffer,
        optimizer_G=optimizer_G,
        optimizer_Disc_A=optimizer_Disc_A,
        optimizer_Disc_B=optimizer_Disc_B,
        sample_interval=hyper.sample_interval
    )