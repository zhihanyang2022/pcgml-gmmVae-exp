import json
import torch.nn as nn
import torch.optim
from collections import OrderedDict 

def conv_sampler(
    in_channels:int, 
    layer_num:int, 
    kernel_nums:tuple, 
    kernel_sizes:tuple, 
    strides:tuple, 
    paddings:tuple,
    final_activation:str=None,
    up_sample:bool=False,
    output_paddings:tuple=None,
    output_type:str='OrderedDict'
)->OrderedDict:
    """
    Return a convolutional sampler (nn.Sequential) with batch-normalizations and leaky ReLUs (for
    down-samplers) or ReLUs (for up-samplers).
    
    The DCGAN paper recommends that kernel sizes should be greater than 3, that strides should be 
    greater than 1, and batch-normalization should be used to guarantee a healthy gradient-flow.
    
    :param up_sample: whether the returned sampler is a up-sampler (default: False)
    """
    
    assert (up_sample and output_paddings is not None) or (not up_sample and output_paddings is None), \
    AssertionError('output_paddings cannot be None when up_sample is True.')
    
    HYPERPARAMS = {
        'conv2d-bias':False,  # set to false because bn introduces biases
        'lrelu-negslope':0.2
    }
    
    # this insight comes from the dcgan paper
    if up_sample: 
        core_layer = nn.ConvTranspose2d
        core_layer_name = 'convtranpose2d'
        activation = nn.ReLU()
    else: 
        core_layer = nn.Conv2d
        core_layer_name = 'conv2d'
        activation = nn.LeakyReLU(HYPERPARAMS['lrelu-negslope'])
        
    layers = OrderedDict([])
    for i in range(layer_num):
        
        if not up_sample:
            layers[f'block{i}-{core_layer_name}'] = core_layer(
                in_channels=in_channels, 
                out_channels=kernel_nums[i], 
                kernel_size=kernel_sizes[i], 
                stride=strides[i],
                padding=paddings[i],
                bias=HYPERPARAMS['conv2d-bias']
            )
        else:
            layers[f'block{i}-{core_layer_name}'] = core_layer(
                in_channels=in_channels, 
                out_channels=kernel_nums[i], 
                kernel_size=kernel_sizes[i], 
                stride=strides[i],
                padding=paddings[i],
                bias=HYPERPARAMS['conv2d-bias'],
                output_padding=output_paddings[i]
            )
            
        layers[f'block{i}-bn'] = nn.BatchNorm2d(kernel_nums[i])
        if i == layer_num - 1:
            if final_activation is not None:
                if final_activation == 'sigmoid':
                    layers[f'block{i}-lrelu'] = nn.Sigmoid()
                elif final_activation == 'relu':
                    layers[f'block{i}-lrelu'] = nn.ReLU()
        else:
            layers[f'block{i}-lrelu'] = activation
        
        in_channels = kernel_nums[i]
        
#     if output_type == 'nn.Sequential':
#         return nn.Sequential(layers)
#     elif output_type == 'OrderedDict':
    return layers  # useful for adding extra layers

class VAEDesign():
    
    def __init__(self, down_sampler_design:dict, up_sampler_design:dict, h_dim:int, z_dim:int, unflatten_out_shape:tuple):
        self.down_sampler_design = down_sampler_design
        self.up_sampler_design = up_sampler_design
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.unflatten_out_shape = unflatten_out_shape
            
    def save_as_json(self, json_fpath:str):
        with open(json_fpath, 'w') as json_f:
            json.dump(self.design_dict, json_f)
                    
    @property
    def design_dict(self):
        return {
            'down_sampler_design':self.down_sampler_design,
            'up_sampler_design':self.up_sampler_design,
            'h_dim':self.h_dim,
            'z_dim':self.z_dim,
            'unflatten_out_shape':self.unflatten_out_shape
        }
    
    @classmethod
    def from_json(cls, json_fpath:str):
        with open(json_fpath, 'r') as json_f:
            design_dict = json.load(json_f)
        return cls(**design_dict)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)  # view(batch_size, flattened_example)

class UnFlatten(nn.Module):
    
    def __init__(self, out_shape:tuple):
        super().__init__()
        self.out_shape = out_shape
    
    def forward(self):
        return input.view(input.size(0), **self.out_shape)

class VAE(nn.Module):
    """Trainable variational auto-encoder implemented in PyTorch."""
    
    def __init__(self, design:VAEDesign, dev:str):
        super(VAE, self).__init__()
        self.dev = dev
        
        # the down-sampler is an OrderedDict of layers
        down_sampler_od = conv_sampler(**design.down_sampler_design)
        down_sampler_od['flatten'] = Flatten()  # append a new layer at the end
        self.encoder = nn.Sequential(down_sampler_od)

        h_dim, z_dim = design.h_dim, design.z_dim
        self.fc1 = nn.Linear(h_dim, z_dim)  # get means
        self.fc2 = nn.Linear(h_dim, z_dim)  # get logvars
        self.fc3 = nn.Linear(z_dim, h_dim)  # process the samples for the up_sampler
        
        # the up-sampler is also an OrderedDict of layers
        up_sampler_od = conv_sampler(**design.up_sampler_design)
        up_sampler_od['unflatten'] = UnFlatten(out_shape=design.unflatten_out_shape)
        up_sampler_od.move_to_end('unflatten', last=False)  # append a new layer at the front
        self.decoder = nn.Sequential(up_sampler_od)

    def reparametrize(self, mu:torch.Tensor, logvar:torch.Tensor)->torch.Tensor:
        """Helper method to self.bottleneck"""
        std = logvar.mul(0.5).exp_()  # logvar to std
        esp = torch.randn(*mu.size())  # number of std
        z = mu + std * esp.to(self.dev).double()  # sample latent vectors
        return z

    def bottleneck(self, h:torch.Tensor)->tuple:
        """Helper method to self.encode"""
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def encode(self, x:torch.Tensor)->tuple:
        """Helper method to self.forward"""
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z:torch.Tensor)->torch.Tensor:
        """Helper method to self.forward"""
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x:torch.Tensor)->tuple:
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

def get_vae_and_opt(design_json_fpath:str, dev:str):
    """Get a trainable VAE and its optimizer."""
    vae = VAE(design=VAEDesign.from_json(design_json_fpath), dev=dev)  # this dev is used in the VAE.reparameterize function
    vae = vae.to(dev).double()  # this dev decides where model parameters are loaded
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    return vae, opt

def load_vae(path:str, design_json_fpath:str, dev:str='cpu'):
    """
    Load trained weights into a VAE architecture.
    
    :param path: the path to the trained weights
    :param design_dict: the Design object of a VAE architecture
    :param dev: where the resulting model would exist (options: 'cpu', 'cuda') (default: 'cpu')
    """
    vae = VAE(design=VAEDesign.from_json(design_json_fpath), dev=dev)
    vae = vae.to(dev).double()
    vae.load_state_dict(torch.load(path, map_location=dev))
    return vae

