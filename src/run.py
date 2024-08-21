import os
import sys
import shutil
# adding project root dir to sys path
project_root_dir = os.path.abspath('..')
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
from src.data.data_loading import load_dataset
from src.data.data_processing import get_data_loaders
from src.data.image_net_dataset import ImageNetDataset
from src.models.cnn_image_classifier import CNNImageClassifier
from src.helpers.tools import (compute_conv_output_size,
                               compute_model_nbr_params,
                               load_yaml_file,
                               initialize_experiment_directories)
from src.procedures.training import train
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import v2
from matplotlib import pyplot as plt
from tqdm import tqdm

YAML_FILE_PATH = './config.yaml'

def main() -> None:

    yaml_info = load_yaml_file(YAML_FILE_PATH)

    print('Training Parameters:')
    for k,v in yaml_info.items():
        print(f'  ---> {k}: {v}')

    log_dir, model_save_dir = \
        initialize_experiment_directories(yaml_info['experiment_name'])

    print(f'\nlog dir: {log_dir}\n')
    
    shutil.copy(YAML_FILE_PATH, os.path.join(model_save_dir, '..'))

    # data loading
    resize = v2.Resize(size=(2**8,) * 2)
    convert_to_float = v2.ToDtype(torch.float32, scale=False)
    transforms_list = [resize, convert_to_float]
    train_loader, val_loader = get_data_loaders(
        train_batch_size=yaml_info['batch_size']['train'],
        val_batch_size=yaml_info['batch_size']['val'],
        shuffle_train=True,
        shuffle_val=False,
        train_transforms=transforms_list,
        val_transforms=transforms_list
        )

    # model definition
    model = CNNImageClassifier(dropout_probability=0.5)
    print(f'\nmodel number of parameters: {compute_model_nbr_params(model):,}')

    # training parameters
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=yaml_info['learning_rate'],
                                weight_decay=yaml_info['weight_decay'])
    scheduler = None
    if yaml_info['scheduler']['use_scheduler']:
        scheduler = ReduceLROnPlateau(
                            optimizer,
                            mode='min',
                            factor=yaml_info['scheduler']['factor'],
                            patience=yaml_info['scheduler']['patience']
                            )
    criteria = torch.nn.CrossEntropyLoss()
    epochs = yaml_info['epochs']

    # training
    model, train_losses, eval_losses, train_accuracies, eval_accuracies = train(model,
                                                                                criteria,
                                                                                optimizer,
                                                                                train_loader,
                                                                                epochs, 
                                                                                evaluation_dataloader=val_loader,
                                                                                early_stopping=True,
                                                                                save_final_model=True,
                                                                                model_save_path=model_save_dir,
                                                                                use_tensorboard=True,
                                                                                tensorboard_log_dir=log_dir
                                                                                )

if __name__ == '__main__':
    main()