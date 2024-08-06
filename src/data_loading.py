'''implementing functions to load the paths of all the images in the ImageNet dataset'''

import os
import random
from collections import namedtuple
from tqdm import tqdm

# Constants
FULL_DATA_PATH_ELEMS = ['/',
                        'home', 
                        'masn', 
                        'datasets', 
                        'imagenet-object-localization-challenge',
                        'ILSVRC', 
                        'Data', 
                        'CLS-LOC'
                        ]
MAPPING_FILE = 'LOC_synset_mapping.txt'
DATA_PATH_PREFIX_SLICE = slice(0, 5)

def load_code_to_label_mapping(data_path_list, data_path_prefix_slice,
                                mapping_file):
    '''Loads the mapping from code to natural language label into a dict and 
    returns it'''
    path_list = data_path_list[data_path_prefix_slice] + [mapping_file]
    mapping_file_full_path = os.path.join(*path_list)
    with open(mapping_file_full_path, 'r', encoding='utf-8') as f:
        file_lines = f.readlines()
        mapping = {}
        for file_line in file_lines:
            cleaned_line = file_line.split()
            mapping[cleaned_line[0]] = ' '.join(cleaned_line[1::])
    return mapping

def load_images_paths(data_path_list, data_split, mapping):
    ImageSample = namedtuple('ImageSample', 'path label_code full_label')
    train_images_path = os.path.join(*data_path_list + [data_split])

    images_list = []

    # load images paths
    if data_split == 'train':
        # TODO: this needs to be a seperate function - train/val/test have 
        # different directory structures --> need a different loading 
        # function for each...

        for directory in tqdm(os.listdir(train_images_path)) :
            directory_full_path = os.path.join(train_images_path, directory)
            for image_name in os.listdir(directory_full_path):
                image_full_path = os.path.join(directory_full_path, image_name)
                image_sample = ImageSample(image_full_path, directory, mapping[directory])
                images_list.append(image_sample)
    else:
        for image_name in tqdm(os.listdir(train_images_path)):
            image_full_path = os.path.join(train_images_path, image_name)
            image_sample = ImageSample(image_full_path, directory, mapping[directory])
            images_list.append(image_sample)


    return images_list

def main():
    '''Returns a list of all the train, validation and test Images'''
    code_to_label = load_code_to_label_mapping(FULL_DATA_PATH_ELEMS,
                                                DATA_PATH_PREFIX_SLICE,
                                                  MAPPING_FILE)
    #print(code_to_label)

    dataset = {}
    for data_split in ['train', 'test', 'val']:
        dataset[data_split] = load_images_paths(FULL_DATA_PATH_ELEMS,
                                                data_split,
                                                code_to_label)

    for k,v in dataset.items():
        print(k, len(v))

if __name__ == '__main__':
    main()
