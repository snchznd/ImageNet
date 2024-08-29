from src.models.blocks import get_conv_block, get_FF_block
from collections import namedtuple
import torch.nn
import torch.nn.functional as F

class CNNImageClassifier(torch.nn.Module):
    def __init__(self, dropout_probability = 0.5):
        super().__init__()

        conv_layers_kwargs = {
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'dilation': 1,
            'bias': True,
            'use_max_pool': True,
            'max_pool_kernel_size': 2,
            'max_pool_stride': 2,
            'use_dropout': True,
            'use_batch_norm': True,
            'dropout_probability': dropout_probability
        }

        # building convolutional layers
        #in_out_channels_dims = [(3, 48), (48, 128), (128, 192), (192, 192), (192, 128)]
        #in_out_channels_dims = [(3, 64), (64, 256), (256, 256), (256, 256), (256, 256)] # best for overfitting so far
        in_out_channels_dims = [(3, 10), (10, 10), (10, 10), (10, 10), (10, 10)]
        self.conv_layers = [
                            elem 
                            for (in_dim, out_dim) in in_out_channels_dims
                            for elem in 
                            get_conv_block(nbr_input_channels=in_dim,
                                                nbr_output_channels=out_dim,
                                                **conv_layers_kwargs)
                            ]

        # building FF layers
        #in_out_dims_FF = [(8 * 8 * 256, 2048), (2048, 2048), (2048, 2048)]
        in_out_dims_FF = [(8 * 8 * 10, 256), (256, 256), (256, 256)]
        self.FF_layers = [elem
                          for in_dim, out_dim in in_out_dims_FF 
                          for elem in get_FF_block(in_dim, out_dim)]
        
        # adding final layer
        self.FF_layers.append(torch.nn.Linear(256, 999))

        # decomposing the architecture into two Sequential objects
        self.convolutions = torch.nn.Sequential(*self.conv_layers)
        self.FF = torch.nn.Sequential(*self.FF_layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.convolutions(x)
        x = x.reshape((batch_size, -1))
        y = self.FF(x)
        return y