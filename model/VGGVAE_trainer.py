import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from .VGGVAE import VGGVAE

class VGGVAE_trainer:

    def __init__(self, channels, latent_size, cnn_outsize, device='cpu'):
        self.vae = VGGVAE(channels, latent_size, cnn_outsize)
        self.vae = self.vae.to(device, non_blocking=True)
        self.device = device

        self.optimizer = optim.Adam(self.vae.parameters())

    def train(self, train_loader):
        self.vae.train()

        mse_loss = 0
        KLD_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device).view(-1, 1, 1080)

            self.optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.vae(data)
            
            mse, KLD = self.vae.loss_function(recon_batch, data, mu, logvar)
            
            loss = mse + KLD
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.vae.parameters(), 1000, norm_type=2)
            
            mse_loss += mse.item()
            KLD_loss += KLD.item()
            
            self.optimizer.step()

        mse_loss /= len(train_loader.dataset)
        KLD_loss /= len(train_loader.dataset)

        return mse_loss, KLD_loss

    def test(self, test_loader):
        self.vae.eval()

        test_loss = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data.to(self.device).view(-1, 1, 1080)

                recon_batch, mu, logvar = self.vae(data)
                
                mse, KLD = self.vae.loss_function(recon_batch, data, mu, logvar)
                
                loss = mse + KLD
                test_loss += loss.item()
                        
        test_loss /= len(test_loader.dataset)

        return test_loss

    def save(self, path):
        torch.save(self.vae.to('cpu').state_dict(), path)
        self.vae.to(self.device)

    def load(self, path):
        self.vae.load_state_dict(torch.load(path))