import torch
import torch.optim as optim
from data_loader import DataSetFromFolder
from model import Generator, Discriminator
from CONFIG import *
from torch.utils.data import DataLoader
from loss import PerceptualLoss
import torch.nn as nn
from pathlib import Path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", str(device))

PATH_G = Path(PATH_MODEL_G)
PATH_D = Path(PATH_MODEL_D)


def load_data():
    train_set = DataSetFromFolder(TRAIN_HR_DIR, TRAIN_LR_DIR, HR_CROP_SIZE, UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_set = DataSetFromFolder(VAL_HR_DIR, VAL_LR_DIR, LR_CROP_SIZE, UPSCALE_FACTOR)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader

def xavier_init_weights(model):
    if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(model.weight)

def train(resume_training = True):

    ### Load data
    train_loader, val_loader = load_data()

    ### Load model
    G = Generator().to(device)
    D = Discriminator().to(device)
    optimizerG = optim.Adam(G.parameters())
    optimizerD = optim.Adam(D.parameters())

    ## Load checkpoint
    if resume_training and PATH_G.exists() and PATH_D.exists() and os.path.getsize(PATH_G) > 0 and os.path.getsize(PATH_D) > 0:
        G, D, optimizerG, optimizerD, last_epoch = load_checkpoint(G,D, optimizerG, optimizerD)
        print("Continue training from last checkpoint...")
    else:
        G.apply(xavier_init_weights)
        D.apply(xavier_init_weights)
        last_epoch = 0


    ## Initialize Loss functions
    criterion = nn.BCELoss()
    generator_criterion = PerceptualLoss().to(device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0



    ### Train
    G.train()
    D.train()

    ### Training loop
    # List to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training loop...")
    # For each epoch
    for epoch in range(last_epoch, NUM_EPOCHS):
        ### Save checkpoint
        save_checkpoint(epoch, G, D, optimizerG, optimizerD)
        for i, data in enumerate(train_loader):
            # data[0] is 1 batch of HR images
            # data[1] is 1 batch of LR images
            #             print(f"\tBatch: {i}/{len(train_hr_loader)//BATCH_SIZE}")
            print(f"\tBatch: {i}/{len(train_loader)}")

            ########################
            # (1) Update Discriminator (D): maximum log(D(x)) + log(1 - D(G(z)))
            ########################
            train_hr_batch = data[0].to(device)
            train_lr_batch = data[1].to(device)

            ## Train with all real HR images
            D.zero_grad()
            real_labels = torch.full(size=(len(train_hr_batch),),
                                     fill_value=real_label, dtype=torch.float, device=device)
            # Forward pass
            output_real = D(train_hr_batch).view(-1)
            # Calculate loss
            errD_real = criterion(output_real, real_labels)
            # Calculate gradients
            errD_real.backward()
            D_x = output_real.mean().item()

            ## Train with all fake SR images
            fake_labels = torch.full(size=(len(train_hr_batch),),
                                     fill_value=fake_label, dtype=torch.float, device=device)
            # Generate fake HR images(SR images)
            sr_image = G(train_lr_batch)
            # Classify all fake batch with D
            output_fake = D(sr_image.detach()).view(-1)  # no gradient will be backproped along this variable
            # Calculate loss of D
            errD_fake = criterion(output_fake, fake_labels)
            # Calculate gradients for D
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            errD = errD_real + errD_fake

            # Update D
            optimizerD.step()

            ################
            # (2) Update Generator (G): minimize log(D(G(z)))
            G.zero_grad()
            # Since D was just updated, perform another forward pass of all fake images batch through D
            output_fake = D(sr_image).view(-1)
            # Calculate loss of G
            # errG = criterion(output_fake, fake_labels)
            errG = generator_criterion(output_fake, sr_image, train_hr_batch)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output_fake.mean().item()
            # Update G
            optimizerG.step()
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, NUM_EPOCHS, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            ## Free up GPA memory
            del train_hr_batch, train_lr_batch, errD, errG, real_labels, fake_labels, output_real, output_fake, sr_image
            torch.cuda.empty_cache()



def save_checkpoint(epoch, G, D, optimizerG, optimizerD):
    checkpoint_G = {
        'epoch': epoch,
        'model': G,
        'model_state_dict': G.state_dict(),
        'optimizer_state_dict': optimizerG.state_dict(),
        # 'loss': lossG
    }
    checkpoint_D = {
        'epoch': epoch,
        'model': D,
        'model_state_dict': D.state_dict(),
        'optimizer_state_dict': optimizerD.state_dict(),
        # 'loss': lossD
    }
    torch.save(checkpoint_G, PATH_G)
    torch.save(checkpoint_D, PATH_D)
    print("Save checkpoint successfully!")


def load_checkpoint(G, D, optimizerG, optimizerD):
    checkpoint_G = torch.load(PATH_G)
    G.load_state_dict(checkpoint_G['model_state_dict'])
    optimizerG.load_state_dict(checkpoint_G['optimizer_state_dict'])
    # loss_G =checkpoint_G['loss']
    checkpoint_D = torch.load(PATH_D)
    D.load_state_dict(checkpoint_D['model_state_dict'])
    optimizerD.load_state_dict(checkpoint_D['optimizer_state_dict'])
    # loss_D = checkpoint_D['loss']
    epoch = checkpoint_G['epoch']

    print('Load checkpoint successfully! Last epoch: '+str(epoch))
    return G,D,optimizerG,optimizerD,epoch





if __name__ == '__main__':
    train()



