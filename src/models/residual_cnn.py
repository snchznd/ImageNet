import torch 
from src.models.blocks import FFResidualLayer, get_conv_block
from src.helpers.tools import get_conv_output_shape


class ResidualCNN(torch.nn.Module):
    def __init__(self,
                nbr_residual_layers : int,
                residual_layers_dim : int,
                nbr_conv_layers : int,
                conv_layers_nbr_channels : int,
                conv_layers_kernel_size : int = (3,3),
                conv_layers_stride : int = (1,1),
                conv_layers_padding : int = (1,1),
                conv_layers_dilation : int = (1,1),
                conv_layers_use_bias : bool = True,
                conv_layers_use_max_pool : bool = True,
                conv_layers_max_pool_kernel_size : int = (2,2),
                conv_layers_max_pool_stride : int = (2,2),
                use_batch_norm : bool = True,
                use_dropout : bool = True,
                dropout_probability : float = .0,
                input_image_width : int = 256,
                input_image_height : int = 256,
                input_image_nbr_channels : int = 3,
                nbr_output_classes : int = 999
                ):

        super().__init__()
        
        if nbr_conv_layers < 1 : 
            raise ValueError(
                'The number of convolutional layers must be at least one.'
                )
        if nbr_residual_layers < 1 : 
            raise ValueError(
                'The number of residual layers must be at least one.'
                )

        # define first convolutional layer
        first_conv_layer = get_conv_block(
            nbr_input_channels=input_image_nbr_channels,
            nbr_output_channels=conv_layers_nbr_channels,
            kernel_size=conv_layers_kernel_size,
            stride=conv_layers_stride,
            padding=conv_layers_padding,
            dilation=conv_layers_dilation,
            bias=conv_layers_use_bias,
            use_max_pool=conv_layers_use_max_pool,
            max_pool_kernel_size=conv_layers_max_pool_kernel_size,
            max_pool_stride=conv_layers_max_pool_stride,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            dropout_probability=dropout_probability
        )

        # define following convolutional layers
        following_conv_layers = [elem
                                 for elem in get_conv_block(
                                    nbr_input_channels=conv_layers_nbr_channels,
                                    nbr_output_channels=conv_layers_nbr_channels,
                                    kernel_size=conv_layers_kernel_size,
                                    stride=conv_layers_stride,
                                    padding=conv_layers_padding,
                                    dilation=conv_layers_dilation,
                                    bias=conv_layers_use_bias,
                                    use_max_pool=conv_layers_use_max_pool,
                                    max_pool_kernel_size=conv_layers_max_pool_kernel_size,
                                    max_pool_stride=conv_layers_max_pool_stride,
                                    use_dropout=use_dropout,
                                    use_batch_norm=use_batch_norm,
                                    dropout_probability=dropout_probability
                                 )
                                 for _ in range(nbr_conv_layers - 1)]

        output_height, output_width = get_conv_output_shape(
            input_height=input_image_height,
            input_width=input_image_width,
            nbr_layers=nbr_conv_layers,
            using_max_pooling=conv_layers_use_max_pool,
            convolutions_kernel_size=conv_layers_kernel_size,
            convolutions_stride=conv_layers_stride,
            convolutions_padding=conv_layers_padding,
            convolutions_dilation=conv_layers_dilation,
            max_pooling_kernel_size=conv_layers_max_pool_kernel_size,
            max_pooling_stride=conv_layers_max_pool_stride
        )

        print(f'{output_height=} | {output_width=}')

        # define convolutional_block
        self.convolutional_block = torch.nn.Sequential(
            *first_conv_layer + following_conv_layers
            )

        # define "conversion" layer
        conv_layers_output_nbr_params = output_height * output_width * conv_layers_nbr_channels
        conversion_layer = torch.nn.Linear(in_features=conv_layers_output_nbr_params,
                                           out_features=residual_layers_dim,
                                           bias=True)

        print(f'{conversion_layer=}')
        # define residual layers
        FF_residual_layers = [FFResidualLayer(
            dim = residual_layers_dim,
            dropout_probability=dropout_probability
        )
        for _ in range(nbr_residual_layers)]

        # define final layer
        final_layer = torch.nn.Linear(in_features=residual_layers_dim,
                                      out_features=nbr_output_classes,
                                      bias=True)

        # define FF residual block
        self.FF_residual_block = torch.nn.Sequential(
            *[conversion_layer] + FF_residual_layers + [final_layer]
        )

    def forward(self, x : torch.Tensor):
        batch_size = x.shape[0]
        x = self.convolutional_block(x)
        x = x.reshape((batch_size, -1))
        x = self.FF_residual_block(x)
        return x