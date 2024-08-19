from typing import Dict, Any
import math
import yaml
import torch

def compute_conv_output_size(h_in, w_in, kernel_size, stride, padding, dilation):
    def numerator(dim_in, idx):
        return dim_in + 2 * padding[idx] - dilation[idx] * (kernel_size[idx] - 1) - 1

    def denominator(idx):
        return stride[idx]

    out_dims = {}
    for dim_name, dim_in, idx in [('H_out', h_in, 0), ('W_out', w_in, 1)]:
        expr = numerator(dim_in, idx) / denominator(idx) + 1
        out_dims[dim_name] = math.floor(expr)
    return out_dims


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