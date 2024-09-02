import torch

class FFResidualLayer(torch.nn.Module):
    def __init__(self, dim, dropout_probability=None):
        super().__init__()
        self.input_dim = dim
        self.output_dim = dim
        self.dropout_probability = dropout_probability
        self.FF_block = torch.nn.Sequential(
            *get_FF_block(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                dropout_probability=self.dropout_probability
                )
            )

    def forward(self, x):
        return x + self.FF_block(x)




def get_FF_block(input_dim,
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
    

def get_conv_block(nbr_input_channels,
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

    if use_dropout and dropout_probability > 0:
        block.append(torch.nn.Dropout(p=dropout_probability))

    if use_max_pool:
        block.append(torch.nn.MaxPool2d(kernel_size=max_pool_kernel_size,
                                    stride=max_pool_stride))

    return block