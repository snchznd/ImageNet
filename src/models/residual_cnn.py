import torch 
from src.models.blocks import FFResidualLayer, get_conv_block


class ResidualCnn(torch.nn.Module):
    def __init__(self,
                nbr_residual_layers : int,
                residual_layers_dim : int,
                nbr_conv_layers : int,
                conv_layers_nbr_channels : int,
                conv_layers_kernel_size : int = 3,
                conv_layers_stride : int = 1,
                conv_layers_padding : int = 1,
                conv_layers_dilation : int = 1,
                conv_layers_use_bias : bool = True,
                conv_layers_use_max_pool : bool = True,
                conv_layers_max_pool_kernel_size : int = 2,
                conv_layers_max_pool_stride : int = 2,
                use_batch_norm : bool = True,
                use_dropout : bool = True,
                dropout_probability : float = .0,
                input_image_width : int = 256,
                input_image_height : int = 256,
                input_image_nbr_channels : int = 3,
                nbr_output_classes : int = 999
                ):
        
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
        following_conv_layers = [get_conv_block(
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
                                 for _ in range(nbr_conv_layers)]

    def __getitem__(self):
        pass