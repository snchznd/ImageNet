import os
import sys
import shutil

# adding project root dir to sys path
project_root_dir = os.path.abspath('..')
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from src.data.data_processing import get_data_loaders
from src.data.image_net_dataset import ImageNetDataset

from torchvision.transforms import v2
from typing import Iterable, Tuple
import torch
from tqdm import tqdm

def update_average(previous_average: torch.Tensor, new_value: torch.Tensor, iteration_number: int) -> torch.Tensor:
    '''
    Updates the running average with a new value.
    '''
    new_average = previous_average + (new_value - previous_average) / iteration_number
    return new_average


def compute_running_average(my_iterable: Iterable[Tuple[torch.Tensor, int]]) -> torch.Tensor:
    '''
    Computes the running average of an ImageNetDatset without loading all values
    into memory.
    '''
    # Initialize the average tensor based on the first element in the iterable
    sample, label = next(iter(my_iterable))
    current_average = torch.zeros_like(sample)
    
    for idx, elem in tqdm(enumerate(my_iterable), desc="Computing Running Average"):
        tensor = elem[0]
        current_average = update_average(current_average, tensor, idx + 1)
    
    return current_average

def get_average_tensors(path='/home/masn/projects/ImageNet/stats'):
    transforms_list = [v2.Resize(size=(2**8,) * 2), v2.ToDtype(torch.float32, scale=False)]
    train_loader, val_loader = get_data_loaders(
        train_batch_size=1,
        val_batch_size=1,
        shuffle_train=False,
        shuffle_val=False,
        train_transforms=transforms_list,
        val_transforms=transforms_list
    )

    print('Computing average for train set...')
    avg_train_tensor = compute_running_average(train_loader)
    torch.save(avg_train_tensor, os.path.join(path, 'avg_train'))
    print('Tensor saved.')

    print('Computing average for test set...')
    avg_val_tensor = compute_running_average(val_loader)
    torch.save(avg_val_tensor, os.path.join(path, 'avg_val'))
    print('Tensor saved.')

if __name__ == '__main__':
    get_average_tensors()
