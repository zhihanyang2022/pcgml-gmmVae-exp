import torch.nn as nn
import torch.optim
from torch.nn import DataParallel

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)  # view(batch_size, flattened_example)

class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)
    
class VAE(nn.Module):
    def __init__(self, dev, nc=17, h_dim=512, z_dim=64):
        super(VAE, self).__init__()
        self.dev = dev
        
        self.encoder = nn.Sequential(
            # input shape: n, 17, 16, 16
            nn.Conv2d(nc, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # output shape: n, 64, 7, 7
            
            # input shape: n, 64, 7, 7
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # output shape: n, 128, 2, 2
            
            # input shape: n, 128, 2, 2
            Flatten()
            # output shape: n, 128 * 2 * 2 = 512
        )

        self.fc1 = nn.Linear(h_dim, z_dim)  # get means
        self.fc2 = nn.Linear(h_dim, z_dim)  # get logvars
        
        self.fc3 = nn.Linear(z_dim, h_dim)  # process the samples

        # similar to generator in DCGAN
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparametrize(self, mu, logvar):
        z = mu
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
    
def get_model(dev, z_dim, nc):
    vae = DataParallel(VAE(dev=dev, z_dim=z_dim, nc=nc))
    vae = vae.to(dev).double()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    return vae, opt

def load_model(path, nc, dev=torch.device('cpu')):
    vae = VAE(nc=nc, dev=dev).double().to(dev)
    vae = DataParallel(vae)
    vae.load_state_dict(torch.load(path, map_location=dev))
    return vae