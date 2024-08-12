'''
Module that implements a class to load dynamically images from the ImageNet 
dataset and apply to them torchvision transforms.
'''

from typing import List, Callable, Optional
import sys
from concurrent.futures import ProcessPoolExecutor
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from tqdm import tqdm

def process_image(img):
    img_shape = read_image(img.path).shape
    img_nbr_channels = img_shape[0]
    if img_nbr_channels == 1:
        return 'bw', img
    elif img_nbr_channels == 3:
        return 'color', img
    else:
        print(f'ERROR: {img.path} has shape {img_shape}', file=sys.stderr)
        return 'error', img

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
        self.initialize_labels_mapping()
        self.filter_out_1D_images()

    def initialize_labels_mapping(self) -> None:
        print('creating string labels to number labels mapping...')
        self.string_to_label_number = {}
        acc = 0
        for *_, label in tqdm(self.dataset):
            if label not in self.string_to_label_number:
                self.string_to_label_number[label] = acc
                acc += 1

    def filter_out_1D_images(self):
        print('separating 1D images and 2D images...')
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_image, self.dataset)))

        self.dataset_color = []
        self.dataset_black_and_white = []
        for result, img in results:
            if result == 'color':
                self.dataset_color.append(img)
            elif result == 'bw':
                self.dataset_black_and_white.append(img)

    def __len__(self) -> int :
        '''
        Returns the length of the dataset.

        Returns:
            The length of the datset, which is equal to the length of 
            the list of images paths we have.
        '''
        return len(self.dataset_color)

    def __getitem__(self, index) -> torch.Tensor:
        '''
        Returns the element at the corresponding index.

        Args:
            index (int): the index of the element to return.

        Returns:
            An image of the dataset, with the torchvision transforms
            applied to it.
        '''
        sample = self.dataset_color[index]
        path = sample.path
        label = self.string_to_label_number[sample.full_label]
        img = read_image(path)
        if self.transforms:
            img = self.transforms(img)

        return img, label
