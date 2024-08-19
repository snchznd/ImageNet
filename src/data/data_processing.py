from src.data.data_loading import load_dataset
from src.data.image_net_dataset import ImageNetDataset
from torch.utils.data import DataLoader

def get_data_loaders(train_batch_size,
                     val_batch_size,
                     shuffle_train,
                     shuffle_val,
                     train_transforms = None,
                     val_transforms = None,
                     num_workers=32
                     ):
    data_splits = load_dataset()
    # ADDED BOUND ON SAMPLES USED FOR DEV PURPOSES -- REMOVE THIS!
    #train_dataset = ImageNetDataset(data_splits['train'][:10_000], train_transforms)
    #val_dataset = ImageNetDataset(data_splits['val'][:10_000], val_transforms)
    train_dataset = ImageNetDataset(data_splits['train'], train_transforms)
    val_dataset = ImageNetDataset(data_splits['val'], val_transforms)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=train_batch_size,
                                   shuffle=shuffle_train,
                                   num_workers=num_workers)
    val_data_loader = DataLoader(dataset=val_dataset,
                                   batch_size=val_batch_size,
                                   shuffle=shuffle_val,
                                   num_workers=num_workers)

    return train_data_loader, val_data_loader
