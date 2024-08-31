import os
from typing import Dict, Any, Tuple
import math
import yaml
import torch

DEFAULT_EXP_PATH = '/home/masn/projects/ImageNet/experiments'

def compute_conv_output_size(h_in : int,
                             w_in : int,
                             kernel_size : Tuple[int, int],
                             stride : Tuple[int, int],
                             padding : Tuple[int, int],
                             dilation : Tuple[int, int]):
    '''
    note that this computation is valid both for the output size of a 
    convolutional layer or for that of a max pooling layer.
    '''
    def numerator(dim_in, idx):
        return dim_in + 2 * padding[idx] - dilation[idx] * (kernel_size[idx] - 1) - 1

    def denominator(idx):
        return stride[idx]

    out_dims = {}
    for dim_name, dim_in, idx in [('H_out', h_in, 0), ('W_out', w_in, 1)]:
        expr = numerator(dim_in, idx) / denominator(idx) + 1
        out_dims[dim_name] = math.floor(expr)
    return out_dims['H_out'], out_dims['W_out']

def get_conv_output_shape(
        input_height: int,
        input_width: int,
        nbr_layers : int,
        using_max_pooling : bool,
        convolutions_kernel_size : Tuple[int, int],
        convolutions_stride : Tuple[int, int],
        convolutions_padding : Tuple[int, int],
        convolutions_dilation : Tuple[int, int],
        max_pooling_kernel_size : Tuple[int, int],
        max_pooling_stride : Tuple[int, int],
        max_pooling_padding : Tuple[int, int] = (0,0),
        max_pooling_dilation : Tuple[int, int] = (1,1)
                     ):
    current_height = input_height
    current_width = input_width
    for _ in range (nbr_layers):
        current_height, current_width = compute_conv_output_size(
            h_in=current_height,
            w_in=current_width,
            kernel_size=convolutions_kernel_size,
            stride=convolutions_stride,
            padding=convolutions_padding,
            dilation=convolutions_dilation,
            ) 
        
        if using_max_pooling:
            current_height, current_width = compute_conv_output_size(
                h_in=current_height,
                w_in=current_width,
                kernel_size=max_pooling_kernel_size,
                stride=max_pooling_stride,
                padding=max_pooling_padding,
                dilation=max_pooling_dilation,
                ) 
    1
    return current_height, current_width
    


def compute_model_nbr_params(model: torch.nn.Module) -> int:
    '''
    Computes the number of trainable parameters of a model and returns it.
    
    Args:
        model (torch.nn.Module): The model for which to compute the number of parameters.

    Returns:
        int: The total number of trainable parameters in the model.
    '''
    nbr_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return nbr_params


def load_yaml_file(file_name : str) -> Dict[str, Any] :
    with open(file=file_name, mode='r') as file:
        try:
            content = yaml.safe_load(file)
            return content
        except yaml.YAMLError as e:
            print(f'Error loading yaml file {file_name}: {e}')
            return {}

def initialize_experiment_directories(experiment_name : str
                                      , experiments_dir_path : str = DEFAULT_EXP_PATH
                                      ) -> Tuple[str, str]:
    '''
    Creates the directories necessary to log experiments and save models and
    returns the paths to the directories created.

    Args:
        experiment_name (str): the name of the experiment
        experiments_dir_path (str): the path to the "experiments" directory 

    Returns:
        tuple: the logging directory and the model saving directory 
    '''
    logs_dir_path = os.path.join(DEFAULT_EXP_PATH, experiment_name, 'logs')
    models_dir_path = os.path.join(DEFAULT_EXP_PATH, experiment_name, 'models')
    for path in logs_dir_path, models_dir_path:
        os.makedirs(path)
    return logs_dir_path, models_dir_path

def main():
    dim = \
    get_conv_output_shape(
        input_height=256,
        input_width=256,
        nbr_layers=5,
        using_max_pooling=True,
        convolutions_kernel_size=(3,3),
        convolutions_stride=(1,1),
        convolutions_padding=(1,1),
        convolutions_dilation=(1,1),
        max_pooling_kernel_size=(2,2),
        max_pooling_stride=(2,2),
        )

    print(f'    ---> {dim}')

if __name__ == '__main__':
    main()