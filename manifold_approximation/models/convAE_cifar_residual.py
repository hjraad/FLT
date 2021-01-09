'''
Contains the convolutional autoencode (AE) model with residual architecture for CIFAR10 dataset 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
           
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()
        
        self.num_hiddens = num_hiddens

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        #
        self._batchnorm_1 = nn.BatchNorm2d(num_hiddens//2)
        #
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        # 
        self._batchnorm_2 = nn.BatchNorm2d(num_hiddens)
        #
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        #
        self._batchnorm_3 = nn.BatchNorm2d(num_hiddens)
        #
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens//2, 
                                 kernel_size=1, stride=1)
        #
        self._batchnorm_4 = nn.BatchNorm2d(num_hiddens//2)
        #
        self._conv_5 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens//16, 
                                 kernel_size=1, stride=1)
        #
        self._batchnorm_5 = nn.BatchNorm2d(num_hiddens//16)
        #
        self.fc1 = nn.Linear(8*8*num_hiddens//16, embedding_dim)
        
    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._batchnorm_1(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._batchnorm_2(x)
        
        x = self._conv_3(x)
        #x = self._batchnorm_3(x)
        x = self._residual_stack(x)
        
        x = self._conv_4(x)
        x = F.relu(x)
        #x = self._batchnorm_4(x)
        
        x = self._conv_5(x)
        x = F.relu(x)
        #x = self._batchnorm_5(x)

        x = x.view(-1, 8*8*self.num_hiddens//16)
        x_comp = self.fc1(x)
        return x_comp
        
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self.num_hiddens = num_hiddens
        
        self._linear_1 = nn.Linear(in_channels, 8*8*num_hiddens//16)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens//16, out_channels=num_hiddens//2, 
                                 kernel_size=1, stride=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=num_hiddens, 
                                 kernel_size=1, stride=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        
        self._conv_trans_4 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_5 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        
        x = self._linear_1(inputs)
        
        x = x.view(-1, self.num_hiddens//16, 8, 8)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        x = self._conv_trans_2(x)
        x = F.relu(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_3(x)
        x = F.relu(x)
        
        x = self._conv_trans_4(x)
        x = F.relu(x)
        
        return self._conv_trans_5(x)


class ConvAutoencoderCIFARResidual(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(ConvAutoencoderCIFARResidual, self).__init__()
        
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens, embedding_dim)
        
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        x_comp = self._encoder(x)
        x_recon = self._decoder(x_comp)

        return x_recon, x_comp


