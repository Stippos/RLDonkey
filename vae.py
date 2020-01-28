import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
    
    def forward(self, x):
        return x.view(x.size(0), self.channels, self.size, self.size)


class Encoder_4_Channels_Small(nn.Module):
    def __init__(self, image_size, linear_input, linear_output, image_channel, image_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            Flatten())
        self.mu = nn.Sequential(
            nn.Linear(int(image_size**2/4)*32, linear_input),
            nn.ReLU(),
            nn.Linear(linear_input, linear_output)
        )
        self.ls = nn.Sequential(
            nn.Linear(int(image_size**2/4)*32, linear_input),
            nn.ReLU(),
            nn.Linear(linear_input, linear_output)
        )

    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)
        log_sigma = self.ls(x)

        return mu, log_sigma

class Decoder_4_Channels_Small(nn.Module):
    def __init__(self, image_size, linear_input, linear_output, image_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(linear_output, linear_input),
            nn.ReLU(),
            nn.Linear(linear_input, 32*int(image_size**2/4)),
            UnFlatten(32, int(image_size/2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=image_channels, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
        
        

class VAE:
    
    def __init__(self, image_size=40, linear_input=250, linear_output=32, lr=0.001, batch_size=64, image_channels=1):

        self.encoder = Encoder_4_Channels_Small(image_size, linear_input, linear_output, image_channels).to(device)
        self.decoder = Decoder_4_Channels_Small(image_size, linear_input, linear_output, image_channels).to(device)
        self.linear_output = linear_output  
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.image_channels = image_channels

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)
        
        self.recon_func = nn.BCELoss()
        self.recon_func.size_average = False

        self.criterion = nn.KLDivLoss()
        self.loss = nn.MSELoss()

        self.images = []
   
    def vae_loss(self, true, pred, mu, log_sigma):
        #print(true.shape)
        #print(pred.shape)
        recon = F.binary_cross_entropy(pred, true, size_average=False)

        KLD = -0.5 * torch.mean(1 + log_sigma - mu.pow(2) - log_sigma.exp())

        return recon, 3*KLD
    
    def sample_z(self, mu, log_sigma):
        std = log_sigma.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).to(device)
        return mu + std * eps


    def update_parameters(self, epochs, n_images):

        sample_size = min(n_images, len(self.images))
        sample = random.sample(self.images, sample_size)
       
        loader = torch.utils.data.DataLoader([torch.Tensor(x).to(device) for x in sample], batch_size = self.batch_size, shuffle=True)
        
        for e in range(epochs):
            train_loss = 0.0
            recon_loss = 0.0
            for i, inputs in enumerate(loader):
                
               
                
                target = inputs.clone() 
                mu, log_sigma = self.encoder(inputs)
                pred = self.decoder(self.sample_z(mu, log_sigma))
                
                recon, kl = self.vae_loss(target, pred, mu, log_sigma)
                loss = recon + kl
                

                self.optimizer.zero_grad()
               # encoder_outputs = self.encoder(inputs)
               # decoder_outputs = self.decoder(encoder_outputs)
                
               # print(inputs.shape)
               # print(encoder_outputs.shape)
               # print(decoder_outputs.shape)
               # loss = self.criterion(decoder_outputs, inputs)
                
                
                loss.backward()
                
                self.optimizer.step()
                
                train_loss += loss.item()
                recon_loss += recon.item()

            print("Train loss: {}, Recon loss: {}".format(train_loss / i, recon_loss / i))

    def update_regular(self):

        loader = torch.utils.data.DataLoader([torch.Tensor(x).to(device) for x in self.images], batch_size = self.batch_size, shuffle=True)
        rl = 0 
        for i, inputs in enumerate(loader):
            
            self.optimizer.zero_grad()
            encoder_output, _ = self.encoder(inputs)
            decoder_output = self.decoder(encoder_output)
            #print(decoder_output)
            #print(inputs)
            loss = self.loss(inputs, decoder_output)
            
            loss.backward()
            self.optimizer.step()

            rl += loss.item()

        print(rl / (i + 1))


    def process_image(self, im):

        if self.image_channels == 1:
            gs_im = np.dot(im[...,:3], [0.299, 0.587, 0.114]) / 255
            return cv2.resize(gs_im, (self.image_size, self.image_size))[np.newaxis, np.newaxis, :]
        else:
            gs_im = im / 255
            return cv2.resize(gs_im, (self.image_size, self.image_size))[np.newaxis, :].reshape(1, 3, self.image_size, self.image_size)
            
            
    def embed(self, image):
        
        im = torch.Tensor(self.process_image(image)).to(device)
        mu, log_sigma = self.encoder.forward(im)
        return mu.detach().cpu().numpy().squeeze()
    
    def decode(self, embedding):
        return self.decoder(torch.FloatTensor(embedding).to(device)).detach().cpu().numpy()
        
    def add_image(self, im):

        self.images.append(self.process_image(im)[0,:])


