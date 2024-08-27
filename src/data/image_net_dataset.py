'''
Module that implements a class to load dynamically images from the ImageNet 
dataset and apply to them torchvision transforms.
'''

#from src.data.statistics import get_average_tensors, get_std_tensors
from typing import List, Callable, Optional
from concurrent.futures import ProcessPoolExecutor
import sys
import os
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
        #print(f'ERROR: {img.path} has shape {img_shape}', file=sys.stderr)
        return 'error', img

class ImageNetDataset(Dataset):
    '''
    A class to dynamically load the images of the ImageNet dataset.
    '''
    def __init__(
            self,
            image_paths : List[str],
            transforms_list : Optional[List[Callable]] = None,
            path : str = '/home/masn/projects/ImageNet/stats',
            split : str = None
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
        self.path = path
        self.split = split
        if transforms_list:
            self.transforms = v2.Compose(transforms_list)
        else:
            self.transforms = None
        self.initialize_labels_mapping()
        self.load_avg_and_std_tensors()
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

    def load_avg_and_std_tensors(self):
        paths = [os.path.join(self.path, x) 
                    for x in ['avg_train', 'avg_val', 'std_train', 'std_val']]
        if not all([os.path.exists(path) for path in paths]):
            raise FileNotFoundError('You need to compute the avg and std'  
                                    'tensors of the train and val splits.')
            #print('Average and std tensors not found. Computing them now.',
            #      file=sys.stderr)
            #get_average_tensors()
            #get_std_tensors()

        self.avg_train = torch.load(paths[0]).squeeze()
        self.avg_val = torch.load(paths[1]).squeeze()
        self.std_train = torch.load(paths[2]).squeeze()
        self.std_val = torch.load(paths[3]).squeeze()


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
        if self.split == 'train':
            img = (img - self.avg_train) / self.std_train
        elif self.split == 'val':
            img = (img - self.avg_val) / self.std_val

        return img, label
