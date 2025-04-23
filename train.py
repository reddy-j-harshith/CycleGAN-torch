import numpy as np
import itertools
import time
import datetime
import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch

from matplotlib.pyplot import figure
from IPython.display import clear_output

from PIL import Image
import matplotlib.image as mping

from utils import *
from CycleGAN import *

import kagglehub
from kagglehub import KaggleDatasetAdapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hyper = Hyperparameters(
    epoch = 0,
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
    n_critic = 5,
    samples_interval = 100,
    num_resi_blocks = 9,
    lambda_cyc = 10.0,
    lambda_id = 5.0
)

# file_path = "./datasets/summer2winter"

file_path = kagglehub.dataset_download("balraj98/summer2winter-yosemite")

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
    plt.figure(figsize = (x, y))
    plt.show()

transforms_ = [
    transforms.Resize((hyper.img_size, hyper.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_loader = DataLoader(
    ImageDataset(root = file_path, mode = hyper.dataset_train_mode, transforms_ = transforms),
    batch_size = hyper.batch_size,
    shuffle = True,
    num_workers = 1,
)

val_loader = DataLoader(
    ImageDataset(root = file_path, mode = hyper.dataset_test_mode, transforms_ = transforms),
    batch_size = 16,
    shuffle = True,
    num_workers = 1,
)

def save_img_samples(batches_done):

    print("batches_done ", batches_done)
    imgs = next(iter(val_loader))

    # There is no gradient calculation, so they are set to 'eval' mode.
    Gen_AB.eval()
    Gen_BA.eval()

    real_A = imgs["A"].to(device)
    fake_B = Gen_AB(real_A)

    real_B = imgs["B"].to(device)
    fake_A = Gen_BA(real_B)

    real_A = make_grid(real_A, nrow = 16, normalize = True)
    real_B = make_grid(real_B, nrow = 16, normalize = True)
    fake_A = make_grid(fake_A, nrow = 16, normalize = True)
    fake_B = make_grid(fake_A, nrow = 16, normalize = True)

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    path = file_path + "/%s.png" % (batches_done)

    save_image(image_grid, path, normalize = False)
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
    Disc_B.paramters(),
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

# FINAL TRAINING FUNCTION

def train(
    Gen_AB:             Generator,
    Gen_BA:             Generator,
    Disc_A:             Discriminator,
    Disc_B:             Discriminator,
    train_dataloader:   DataLoader,
    n_epochs:           int,
    identity_loss:           torch.nn.L1Loss,
    cycle_loss:              torch.nn.L1Loss,
    GAN_loss:                torch.nn.MSELoss,
    lamdba_cyc:         float,
    lambda_id:          float,
    fake_A_Buffer:      ReplayBuffer,
    fake_B_Buffer:      ReplayBuffer,
    clear_output,
    optimizer_G:        torch.optim.Adam,
    optimizer_Disc_A:   torch.optim.Adam,
    optimizer_Disc_B:   torch.optim.Adam,
    Tensor:             torch.Tensor,
    sample_interval:    int,
):
    
    prev_time = time.time()

    for epoch in range(hyper.epoch, n_epochs):
        for i, batch in enumerate(tqdm(train_dataloader, desc = "epoch")):

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

            loss_G = loss_GAN + lamdba_cyc * loss_cycle + lambda_id * loss_identity
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
