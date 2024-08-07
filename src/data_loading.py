'''
Implementing functions to load the paths of all the images in the train,
validation and test splits of the ImageNet dataset. 

IMPORTANT: Note that this implementation assumes you downloaded the data from 
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
and placed the unzipped imagenet-object-localization-challenge directory in
the '/home/<user_name>/datasets/' directory. This can be modified by modifying
the USER_NAME variable and FULL_DATA_PATH_ELEMS list.
'''

# Imports
import os
from collections import namedtuple
from typing import List, Dict
from tqdm import tqdm

# Constants
USER_NAME = 'masn'
FULL_DATA_PATH_ELEMS = ['/',
                        'home', 
                        USER_NAME,
                        'datasets',
                        'imagenet-object-localization-challenge',
                        'ILSVRC', 
                        'Data',
                        'CLS-LOC'
                        ]
MAPPING_FILE = 'LOC_synset_mapping.txt'
DATA_PATH_PREFIX_SLICE = slice(0, 5)
ImageSample = namedtuple('ImageSample', 'path label_code full_label')

# Functions
def load_code_to_label_mapping(
        data_path_list : List[str],
        data_path_prefix_slice : slice,
        mapping_file : str
    ) -> Dict[str, str]:
    '''
    Loads the mapping from code labels to natural language label into a dict 
    and returns it.

    Args:
        data_path_list (List[str]): directories forming the absolute path to 
            the train, test and val directories
        data_path_prefix_slice (slice): slice indicating the elements of
            data_path_list corresponding to the path to the top-most level
            of the data directory.
        mapping (dict): mapping from label code to actual english label.

    Returns:
        dict: mapping from label codes to label names
    '''
    path_list = data_path_list[data_path_prefix_slice] + [mapping_file]
    mapping_file_full_path = os.path.join(*path_list)
    with open(mapping_file_full_path, 'r', encoding='utf-8') as f:
        file_lines = f.readlines()
        mapping = {}
        for file_line in file_lines:
            cleaned_line = file_line.split()
            mapping[cleaned_line[0]] = ' '.join(cleaned_line[1::])
    return mapping

def load_train_images_paths(
        train_folder_path : str,
        mapping : Dict[str, str]
    ) -> List[ImageSample]:
    '''
    Loads the absolute paths of all the train images as well as their
    corresponding labels and returns them as a list of ImageSample.

    Args:
        train_folder_path (str): the path to the 'train' folder.
        mapping (dict): mapping from label code to actual english label.

    Returns:
        list: The list containing ImageSample describing the paths of the 
            images and their label for the train split of the data.
    '''
    images_list = []

    for directory in tqdm(os.listdir(train_folder_path)) :
        directory_full_path = os.path.join(train_folder_path, directory)
        for image_name in os.listdir(directory_full_path):
            image_full_path = os.path.join(directory_full_path, image_name)
            image_sample = ImageSample(image_full_path, directory, mapping[directory])
            images_list.append(image_sample)
    return images_list

def load_images_paths(
        data_path_list : List[str],
        data_split : str,
        mapping : Dict[str, str]
    ) -> List[ImageSample]:
    '''
    Loads the image paths and labels corresponding to the train, test or
    validation splits of the data and returns them as a list of ImageSample.

    Args:
        data_path_list (List[str]): directories forming the absolute path to 
            the train, test and val directories
        data_split (str): 'train', 'val' or 'test'
        mapping (dict): mapping from label code to actual english label.

    Returns:
        list: The list containing ImageSample describing the paths of the 
            images and their label for the data split passed as argument.
    '''
    data_split_path = os.path.join(*data_path_list + [data_split])
    if data_split == 'train':
        images_list = load_train_images_paths(data_split_path, mapping)
    else:
        pass

    return images_list

def main() -> Dict[str, List[ImageSample]]:
    '''
    Creates and returns a dict containing the absolute paths to all the train,
    validation and test Images.

    Returns:
        dict: A dict containing the 'train', 'val' and 'test' keys with the
            corresponding values being lists of ImageSample instances 
            representing the absolute paths and labels of the images.
    '''
    code_to_label = load_code_to_label_mapping(FULL_DATA_PATH_ELEMS,
                                                DATA_PATH_PREFIX_SLICE,
                                                MAPPING_FILE)

    dataset = {}
    for data_split in ['train', 'test', 'val']:
        dataset[data_split] = load_images_paths(FULL_DATA_PATH_ELEMS,
                                                data_split,
                                                code_to_label)

    for k,v in dataset.items():
        if v:
            print(k, len(v))

    return dataset

if __name__ == '__main__':
    main()
