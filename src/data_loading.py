'''
Implementing functions to load the paths of all the images in the train and
validation splits of the ImageNet dataset. 

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
import xml.etree.ElementTree as ET
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
XML_FILE_NAME_IDX = 1
XML_OBJ_IDX = 5
XML_NAME_IDX = 0
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

def load_validation_images_paths(
        data_path_list : List[str],
        data_path_prefix_slice : slice,
        mapping : Dict[str, str]
    ) -> List[ImageSample]:
    '''
    Loads the absolute paths of all the validation images as well as their
    corresponding labels and returns them as a list of ImageSample.

    Args:
        data_path_list (List[str]): directories forming the absolute path to 
            the train, test and val directories
        data_path_prefix_slice (slice): slice indicating the elements of
            data_path_list corresponding to the path to the top-most level
            of the data directory.
        mapping (dict): mapping from label code to actual english label.

    Returns:
        list: The list containing ImageSample describing the paths of the 
            images and their label for the validation split of the data.
    '''
    validation_folder_path = os.path.join(*data_path_list + ['val'])
    path_end = ['ILSVRC', 'Annotations', 'CLS-LOC', 'val']
    data_labels_path = os.path.join(*data_path_list[data_path_prefix_slice],
                                    *path_end)
    xml_annotation_files = os.listdir(data_labels_path)

    # get labels
    name_to_code : Dict[str, str] = {}
    for file_name in tqdm(xml_annotation_files):
        full_file_path = os.path.join(data_labels_path, file_name)
        tree = ET.parse(full_file_path)
        tree_root = tree.getroot()
        file_name = tree_root[XML_FILE_NAME_IDX].text
        label_code = tree_root[XML_OBJ_IDX][XML_NAME_IDX].text
        name_to_code[file_name] = label_code

    # get images
    images_list = []
    for image_name in tqdm(os.listdir(validation_folder_path)):
        image_full_path = os.path.join(validation_folder_path, image_name)
        image_label_code = name_to_code[image_name[:-5]]
        image_sample = ImageSample(image_full_path, image_label_code, mapping[image_label_code])
        images_list.append(image_sample)

    return images_list

def load_images_paths(
        data_path_list : List[str],
        data_path_prefix_slice : slice,
        data_split : str,
        mapping : Dict[str, str]
    ) -> List[ImageSample]:
    '''
    Loads the image paths and labels corresponding to the train or
    validation splits of the data and returns them as a list of ImageSample.

    Args:
        data_path_list (List[str]): directories forming the absolute path to 
            the train, test and val directories
        data_path_prefix_slice (slice): slice indicating the elements of
            data_path_list corresponding to the path to the top-most level
            of the data directory.
        data_split (str): 'train' or 'val'
        mapping (dict): mapping from label code to actual english label.

    Returns:
        list: The list containing ImageSample describing the paths of the 
            images and their label for the data split passed as argument.
    '''
    data_split_path = os.path.join(*data_path_list + [data_split])
    if data_split == 'train':
        print('loading train set...')
        images_list = load_train_images_paths(data_split_path, mapping)
    elif data_split == 'val':
        print('loading validation set...')
        images_list = load_validation_images_paths(data_path_list, data_path_prefix_slice, mapping)
    else:
        raise ValueError('data split must be "train" or "val"')

    return images_list

def load_dataset() -> Dict[str, List[ImageSample]]:
    '''
    Creates and returns a dict containing the absolute paths to all the train
     and validation Images.

    Returns:
        dict: A dict containing the 'train' and 'val' keys with the
            corresponding values being lists of ImageSample instances 
            representing the absolute paths and labels of the images.
    '''
    code_to_label = load_code_to_label_mapping(FULL_DATA_PATH_ELEMS,
                                                DATA_PATH_PREFIX_SLICE,
                                                MAPPING_FILE)

    dataset = {}
    for data_split in ['train', 'val']:
        dataset[data_split] = load_images_paths(FULL_DATA_PATH_ELEMS,
                                                DATA_PATH_PREFIX_SLICE,
                                                data_split,
                                                code_to_label)

    return dataset
