'''
Module that implements a class to load dynamically images from the ImageNet 
dataset and apply to them torchvision transforms.
'''

from typing import List, Callable, Optional
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

class ImageNetDataset(Dataset):
    '''
    A class to dynamically load the images of the ImageNet dataset.
    '''
    def __init__(
            self,
            image_paths : List[str],
            transforms_list : Optional[List[Callable]] = None
            ) -> None:
        '''
        Initializes the dataset.

        Args:
            image_paths (list): a list containing the path of the images
                that are part of the datset.
            transforms_list (list): a list containing the torchvision
                transforms to apply to the image.
        '''
        self.dataset = image_paths
        if transforms_list:
            self.transforms = v2.Compose(transforms_list)
        else:
            self.transforms = None

    def __len__(self) -> int :
        '''
        Returns the length of the dataset.

        Returns:
            The length of the datset, which is equal to the length of 
            the list of images paths we have.
        '''
        return len(self.dataset)

    def __getitem__(self, index) -> torch.Tensor:
        '''
        Returns the element at the corresponding index.

        Args:
            index (int): the index of the element to return.

        Returns:
            An image of the dataset, with the torchvision transforms
            applied to it.
        '''
        path = self.dataset[index].path
        img = read_image(path)
        if self.transforms:
            img = self.transforms(img)
        return img
