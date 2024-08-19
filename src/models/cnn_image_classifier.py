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
        in_out_channels_dims = [(3, 48), (48, 128), (128, 192), (192, 192), (192, 128)]
        self.conv_layers = [
                            elem 
                            for (in_dim, out_dim) in in_out_channels_dims
                            for elem in 
                            self.get_conv_block(nbr_input_channels=in_dim,
                                                nbr_output_channels=out_dim,
                                                **conv_layers_kwargs)
                            ]

        # building FF layers
        in_out_dims_FF = [(8 * 8 * 128, 2048), (2048, 2048)]
        self.FF_layers = [elem
                          for in_dim, out_dim in in_out_dims_FF 
                          for elem in self.get_FF_block(in_dim, out_dim)]
        
        # adding final layer
        self.FF_layers.append(torch.nn.Linear(2048, 999))

        # decomposing the architecture into two Sequential objects
        self.convolutions = torch.nn.Sequential(*self.conv_layers)
        self.FF = torch.nn.Sequential(*self.FF_layers)

    def get_FF_block(self,
                     input_dim,
                     output_dim,
                     dropout_probability=None):
        layers = [
            torch.nn.Linear(in_features=input_dim,
                            out_features=output_dim),
            torch.nn.BatchNorm1d(num_features=output_dim),
            torch.nn.ReLU(),
                ]
        if dropout_probability:
            layers.append(torch.nn.Dropout(p=dropout_probability))

        return layers
        

    def get_conv_block(self,
                       nbr_input_channels,
                       nbr_output_channels,
                       kernel_size,
                       stride,
                       padding,
                       dilation,
                       bias,
                       use_max_pool,
                       max_pool_kernel_size,
                       max_pool_stride,
                       use_dropout,
                       use_batch_norm,
                       dropout_probability=0.5):
        block = []
        convolution = torch.nn.Conv2d(in_channels=nbr_input_channels,
                                      out_channels=nbr_output_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=bias)
        block.append(convolution)

        if use_batch_norm:
            block.append(torch.nn.BatchNorm2d(num_features=nbr_output_channels))

        block.append(torch.nn.ReLU())

        if use_dropout:
            block.append(torch.nn.Dropout(p=dropout_probability))

        if use_max_pool:
            block.append(torch.nn.MaxPool2d(kernel_size=max_pool_kernel_size,
                                        stride=max_pool_stride))

        return block

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.convolutions(x)
        x = x.reshape((batch_size, -1))
        y = self.FF(x)
        return y