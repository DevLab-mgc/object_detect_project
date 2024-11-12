from torchvision.transforms import v2 as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import ObjectCocoAlbumention
from torch.utils.data import DataLoader, Subset
from torch import randperm
import utils
import torch
from torchvision import transforms
#from torchvision.transforms.transforms import Compose
import numpy as np
import random

max_size=512
np.random.seed(42)
def get_albumentations_transform(train):
    transform_pipe = []
    #A.LongestMaxSize(256),
    augs = [A.BBoxSafeRandomCrop(p=0.5),  A.HorizontalFlip(p=0.5), 
            A.RandomScale(p=0.5, scale_limit=0.5), A.RandomBrightnessContrast(), 
            A.RandomGamma(),A.CLAHE(),
            A.LongestMaxSize(max_size)]
    if train:
        for aug_i in augs:
            transform_pipe.append(aug_i)

    transform_pipe.append(A.Normalize())#A.Resize(256, 256, always_apply=True))
    transform_pipe.append(A.ToFloat(always_apply=True))
    transform_pipe.append(ToTensorV2())
    return A.Compose(transform_pipe, bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))



def get_data_loaders(train_dataset_path, 
                    val_dataset_path,
                    train_annFile,
                    val_annFile, 
                    train_batch_size, 
                    val_batch_size, 
                    bbox_in_format='coco', 
                    bbox_out_format='xyxy',
                    processor_type='albumentations',
                    testing = False):

    train_transform = get_albumentations_transform(train=True)
    val_transform = get_albumentations_transform(train=False)

    
    #img_folder, ann_file, processor=None, bbox_format='coco', binary_task=True, train=True
    train_dataset = ObjectCocoAlbumention(img_folder=train_dataset_path,
                                    ann_file=train_annFile,
                                    processor=train_transform,
                                    bbox_in_format=bbox_in_format,
                                    bbox_out_format=bbox_out_format,
                                    binary_task=True
                                        )
    
    val_dataset = ObjectCocoAlbumention(img_folder=val_dataset_path,
                                         ann_file=val_annFile,
                                         processor=val_transform,
                                    bbox_in_format=bbox_in_format,
                                    bbox_out_format=bbox_out_format,
                                           binary_task=True
                                         )

    #if 'consolidated_coco' in train_annFile:
    if testing:
        print("----------MANUAL INDEX SPLITTING------------")
        ## split the dataset in train and test set
        indices = randperm(len(train_dataset)).tolist()
        val_indices = randperm(len(val_dataset)).tolist()
        
        train_dataset = Subset(train_dataset, indices[:5])
        val_dataset = Subset(val_dataset, val_indices[:5])

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=utils.collate_fn)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=val_batch_size, 
                            shuffle=False,
                            num_workers=4, 
                            collate_fn=utils.collate_fn)
    
    return train_loader, val_loader
