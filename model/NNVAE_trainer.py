import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from .NNVAE import NNVAE

import torch_optimizer as optim_

class NNVAE_trainer:

    def __init__(self, data_size, hidden_sizes, latent_size, activation_function='relu', bn_flag=False, adaBelief=False, device='cpu'):
        self.vae = NNVAE(data_size, hidden_sizes, latent_size, activation_function, bn_flag)
        self.vae = self.vae.to(device)
        self.device = device

        if adaBelief:
            self.optimizer = optim_.AdaBelief(self.vae.parameters())
        else:
            self.optimizer = optim.Adam(self.vae.parameters())

    def train(self, train_loader, k=1.0):
        self.vae.train()

        mse_loss = 0
        KLD_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)

            self.optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.vae(data)
            
            mse, KLD = self.vae.loss_function(recon_batch, data, mu, logvar)
            
            KLD = k*KLD
            loss = mse + KLD
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.vae.parameters(), 1000, norm_type=2)
            
            mse_loss += mse.item()
            KLD_loss += KLD.item()
            
            self.optimizer.step()

        mse_loss /= len(train_loader.dataset)
        KLD_loss /= len(train_loader.dataset)

        return mse_loss, KLD_loss

    def test(self, test_loader, k=1.0):
        self.vae.eval()

        test_loss = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data.to(self.device)

                recon_batch, mu, logvar = self.vae(data)
                
                mse, KLD = self.vae.loss_function(recon_batch, data, mu, logvar)
                KLD = k*KLD
                
                loss = mse + KLD
                test_loss += loss.item()
                        
        test_loss /= len(test_loader.dataset)

        return test_loss


    def save(self, path):
        torch.save(self.vae.to('cpu').state_dict(), path)
        self.vae.to(self.device)

    def load(self, path):
        self.vae.load_state_dict(torch.load(path))