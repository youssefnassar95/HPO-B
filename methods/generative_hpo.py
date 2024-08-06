"""
Author: Youssef Nassar
Date: 29 December 2023
"""

import random
import numpy as np
import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenerativeHPO:

    def __init__(self,num_tokens=17,dim_model=32,n_batches=100,batch_size=16):

        print("Using Generative HPO method...")
        self.num_tokens = num_tokens
        self.dim_model = dim_model

        self.n_batches = n_batches
        self.batch_size = batch_size

        self.embedding = MlpEmbedding(input_size=self.num_tokens, output_size=self.dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_model, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.model_vae = VAE(feat_dim=self.num_tokens, model_dim=self.dim_model, 
                             output_dim=self.num_tokens-1).to(device)
        self.opt_vae = torch.optim.Adamax(self.model_vae.parameters(), lr=0.01)
        self.loss_fn_vae = nn.MSELoss()

    def observe_and_suggest(self, X_obs, y_obs):
        X_obs = torch.tensor(X_obs).to(device)
        y_obs = torch.tensor(y_obs).to(device)

        for i in range(self.n_batches):
            X_b = []
            y_b = []
            x_sample = []
            I = []
            c_size = torch.randint(1, X_obs.shape[0],(1,))
            sample_idx = torch.randint(1, X_obs.shape[0],(1,))
            
            for j in range(self.batch_size):
                sample_idx = torch.randint(1, X_obs.shape[0],(1,))
                # c_indices = torch.randint(0, X_obs.shape[0],(c_size,))
                c_indices = random.sample(range(0, X_obs.shape[0]), c_size)
                x_sample.append(X_obs[sample_idx])
                X_b.append(X_obs[c_indices])
                y_b.append(y_obs[c_indices])
                if y_obs[sample_idx]>y_obs[c_indices].max():
                    I.append(1)
                else:
                    I.append(0)
            loss = self.fit(x_sample,I, X_b, y_b)

        self.transformer_encoder.eval()
        self.model_vae.eval()
        src = torch.concat((X_obs, y_obs), dim=1).unsqueeze(0)
        src = src.permute(1,0,2)
        src = self.embedding(src.float()) * math.sqrt(self.dim_model)
        out = self.transformer_encoder(src)
        out = out.mean(dim=1)
        I = torch.ones([1, 1])
        z = torch.mean(torch.rand(10,2), dim=0).unsqueeze(0)

        conca_dec = torch.concat((z, I, out), dim=1).float()
        x_new = self.model_vae.decode(conca_dec)
        x_new = x_new.detach().numpy()
        return x_new

    def fit(self,x_sample,I, X_b, y_b):
        """
        This is the fitting function for the VAE training.
        :param x_sample: the sampled x with dim (B,F), B=batch size, F=number of features
        :param I: The improvement flag with dim (B,1)
        :X_b: the sequence of x sampled from the history to create subset C dim(B,C,F)
        :y_b: the sequence of y sampled from the history to create subset C dim(B,C,1)
        """
        self.transformer_encoder.train()
        self.model_vae.train()

        src = torch.concat((torch.stack(X_b),torch.stack(y_b)), dim=2)
        self.embedding = self.embedding.float()
        self.model_vae.float()
        self.loss_fn_vae.float()

        src = src.permute(1,0,2)
        src = self.embedding(src.float()) * math.sqrt(self.dim_model)
        out = self.transformer_encoder(src)
        out = out.mean(dim=1)

        x_sample = torch.stack(x_sample).squeeze(1)
        I = torch.tensor(I).unsqueeze(1)
        x_hat, mean, logvar = self.model_vae(x_sample, I, out)
        bce_loss = self.loss_fn_vae(x_hat.float(), x_sample.float())
        loss_vae = self.model_vae.final_loss(bce_loss, mean, logvar)

        self.opt_vae.zero_grad()

        loss = loss_vae
        loss.backward()
        self.opt_vae.step()

        return loss

class MlpEmbedding(nn.Module):
    def __init__(self, input_size,  output_size, hidden_dim=32):
        super(MlpEmbedding, self).__init__()

        self.embedding_layer=nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_size),
            nn.LeakyReLU(0.2)
            )
        self.out_size = output_size
    def forward(self,x):
        # x will be in shape (C,B,17) we want to make it in shape (B*C,17)
        C_size = x.shape[0]
        batch_size = x.shape[1]
        features_size = x.shape[2]
        x = x.reshape(batch_size*C_size,features_size)
        x = self.embedding_layer(x)
        x = x.reshape(batch_size, C_size, self.out_size)
        return x


class VAE(nn.Module):

    def __init__(self, feat_dim, model_dim, output_dim, hidden_dim=32, latent_dim=64, z_dim=2, device=device):
        super(VAE, self).__init__()

        # encoder
        self.input_dim = feat_dim + model_dim
        self.dec_input_dim = model_dim + z_dim + 1
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, z_dim)
        self.logvar_layer = nn.Linear(latent_dim, z_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.dec_input_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, I, out_tr):
        """
        This is the forward pass for the VAE.
        It takes the inputs, concatenate them and pass it to the VAE, return the 
        reconstructed x_hat alongside mu and sigma
        :param x: the sampled x with dim (B,F), B=batch size, F=number of features
        :param I: The improvement flag with dim (B,1)
        :param out_tr: This is the output from the Transformer with dim(B,D), 
        C=sequence length, D=Transformer model dimension
        """
        conca = torch.concat((x, I, out_tr), dim=1).float()
        mean, logvar = self.encode(conca)
        z = self.reparameterization(mean, logvar)
        conca_dec = torch.concat((z, I, out_tr), dim=1).float()
        x_hat = self.decode(conca_dec)
        return x_hat, mean, logvar
    
    def final_loss(self, rec_loss, mu, logvar):
        """
        This function will add the reconstruction loss and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param rec_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        REC = rec_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return REC + KLD
