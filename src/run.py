import os
import sys
import random
from PIL import Image

## adding project root dir to sys path
project_root_dir = os.path.abspath('..')
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from src.data.data_loading import load_dataset
from src.data.data_processing import get_data_loaders
from src.data.image_net_dataset import ImageNetDataset
from src.models.cnn_image_classifier import CNNImageClassifier
from src.helpers.tools import compute_conv_output_size, compute_model_nbr_params
from src.procedures.training import train

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import v2
from matplotlib import pyplot as plt
from tqdm import tqdm

# data loading
resize = v2.Resize(size=(2**8,) * 2)
convert_to_float = v2.ToDtype(torch.float32, scale=False)
transforms_list = [resize, convert_to_float]
train_loader, val_loader = get_data_loaders(train_batch_size=150,
                                            val_batch_size=150,
                                            shuffle_train=True,
                                            shuffle_val=False,
                                            train_transforms=transforms_list,
                                            val_transforms=transforms_list
                                           )

# model definition
model = CNNImageClassifier(dropout_probability=0.5)
print(f'  ---> model number of parameters: {compute_model_nbr_params(model):,}')

# training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
criteria = torch.nn.CrossEntropyLoss()
epochs = 2

# training
model, train_losses, eval_losses, train_accuracies, eval_accuracies = train(model,
                                                                            criteria,
                                                                            optimizer,
                                                                            train_loader,
                                                                            epochs, 
                                                                            evaluation_dataloader=val_loader,
                                                                            early_stopping=True,
                                                                            save_final_model=True,
                                                                            model_save_path='/home/masn/projects/ImageNet/experiments/models/experiment_0',
                                                                            use_tensorboard=True,
                                                                            tensorboard_log_dir='/home/masn/projects/ImageNet/experiments/logs/experiment_0'
                                                                            )