import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit

# class Encoder(nn.Module):
class Encoder(jit.ScriptModule):
    def __init__(self, x_size, hidden_sizes, latent_size, activation_function='relu', bn_flag=False):
        super().__init__()
        self.x_size  = x_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.bn_flag = bn_flag

        self.act_fn = getattr(F, activation_function)

        layer = [nn.Linear(x_size, hidden_sizes[0])]
        for i in range(len(hidden_sizes)-1):
            layer.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.layers = nn.ModuleList(layer)

        if bn_flag:
            bn_layer = [nn.BatchNorm1d(hidden_sizes[0])]
            for i in range(len(hidden_sizes)-1):
                bn_layer.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            self.bn_layers = nn.ModuleList(bn_layer).to('cuda')

        self.fc_mu    = nn.Linear(hidden_sizes[-1], latent_size)
        self.fc_sigma = nn.Linear(hidden_sizes[-1], latent_size)

    def forward(self, x):
        h = x.view(-1, self.x_size)

        if self.bn_flag:
            for layer, bn_layer in zip(self.layers, self.bn_layers):
                h = self.act_fn(bn_layer(layer(h)))
        else:
            for layer in self.layers:
                h = self.act_fn(layer(h))

        mu     = self.fc_mu(h)
        logvar = self.fc_sigma(h)
        return mu, logvar

# class Decoder(nn.Module):
class Decoder(jit.ScriptModule):
    def __init__(self, x_size, hidden_sizes, latent_size, activation_function='relu', bn_flag=False):
        super().__init__()
        self.x_size  = x_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.bn_flag = bn_flag

        self.act_fn = getattr(F, activation_function)

        layer = [nn.Linear(latent_size, hidden_sizes[-1])]
        for i in reversed(range(len(hidden_sizes)-1)):
            layer.append(nn.Linear(hidden_sizes[i+1], hidden_sizes[i]))
        self.layers = nn.ModuleList(layer)

        if bn_flag:
            bn_layer = [nn.BatchNorm1d(hidden_sizes[-1])]
            for i in reversed(range(len(hidden_sizes)-1)):
                bn_layer.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.bn_layers = nn.ModuleList(bn_layer).to('cuda')

        self.fc = nn.Linear(hidden_sizes[0], x_size)

    def forward(self, z): 
        h = z.view(-1, self.latent_size)

        if self.bn_flag:
            for layer, bn_layer in zip(self.layers, self.bn_layers):
                h = self.act_fn(bn_layer(layer(h)))
        else:
            for layer in self.layers:
                h = self.act_fn(layer(h))

        x = torch.sigmoid(self.fc(h))

        return x

# class VAE(nn.Module):
class NNVAE(jit.ScriptModule):
    def __init__(self, x_size, hidden_sizes, latent_size, activation_function='relu', bn_flag=False):
        super().__init__()
        self.x_size  = x_size
        self.encoder = Encoder(x_size, hidden_sizes, latent_size, activation_function, bn_flag)
        self.decoder = Decoder(x_size, hidden_sizes, latent_size, activation_function, bn_flag)

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, self.x_size))
        sigma = logvar.mul(0.5).exp()
        eps   = torch.randn_like(sigma)

        if self.training:
            z = eps.mul(sigma).add_(mu) # mu + sigma^(1/2) * eps
        else:
            z = mu

        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):

        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_size), reduction='sum')
        # BCE = F.mse_loss(recon_x, x, size_average=False)
 
        # 0.5*(1 + log(sigma^2) - mu^2 - sigma^2) 
        # 実装ではsigmaがマイナスになるとlogsigmaを求められないためか、2*logsigmaをlogvarと置いて
        # KL距離を0.5*(mu^2 + exp(logvar) −logvar − 1) とする記述が主流?
        # https://qiita.com/nishiha/items/2264da933504fbe3fc68

        KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        # KLD = 0

        return BCE, KLD

if __name__ == '__main__':

    from torchsummary import summary

    # bn_flag=False
    bn_flag=True

    hiddens = [90]

    vae = VAE(x_size=180, hidden_sizes=hiddens, latent_size=18, bn_flag=bn_flag)
    # summary(vae, torch.zeros((1, 180)))
    vae(torch.zeros((10, 180)))

    hiddens = [90, 45]

    vae = VAE(x_size=180, hidden_sizes=hiddens, latent_size=18, bn_flag=bn_flag)
    # summary(vae, torch.zeros((1, 180)))
    vae(torch.zeros((10, 180)))

    # hiddens = [90, 60, 30]

    # vae = VAE(x_size=180, hidden_sizes=hiddens, latent_size=18, bn_flag=bn_flag)
    # summary(vae, torch.zeros((1, 180)))