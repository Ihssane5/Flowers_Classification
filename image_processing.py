from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt


def Load_DataSet(data_dir,transformations,batch_size, shuffle):
    """
    Processes the dataset located in the specified directory.

    Args:
        data_dir (str): The path to the dataset directory. This directory should contain 
                        subdirectories named 'train', 'test', and 'valid', each with their 
                        respective data.
        transformations (dict): Dictionary of your transformations for the 'train', 'test', and 'valid'
        batch_size (int): the batch size for loading the data
        shuffle (bool): a boolean that specify if the data will be shuffled  during Loading process

    Returns:
        dataloaders (dict) : Dictionary for the dataloders for the 'train', 'test', and 'valid'
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(root=train_dir,transform = transformations['train'])
    test_datasets = datasets.ImageFolder(root=test_dir,transform = transformations['test'])
    valid_datasets = datasets.ImageFolder(root=valid_dir, transform = transformations['valid'])


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = DataLoader(train_datasets, batch_size = batch_size, shuffle = shuffle)
    test_dataloaders = DataLoader(test_datasets, batch_size = batch_size,)
    valid_dataloaders = DataLoader(valid_datasets,batch_size = batch_size, shuffle = shuffle)

    dataloders = {
        'train' : train_dataloaders,
        'valid' : valid_dataloaders,
        'test' : test_dataloaders,
    }
    return dataloders

def Process_Image(image):
    """Processes the image with a specific path.

    Args:
        image (str): The path of the image to use for inference
    Returns:
        tnr_img (torch.tensor) : Tensor representation of the image
    """
    pil_img = Image.open(image)
    height, width  = pil_img.height, pil_img.width
    ratio = height / width
    if (height <= width):
        height = 256
        width = height / ratio
    else:
        width = 256
        height = int(ratio * width)
    (width, height) = (int(width), int(height))
    resize_img = pil_img.resize((width, height))
    left = (resize_img.width - 244) / 2
    top = (resize_img.height - 244) / 2
    right = left + 244
    bottom = top + 244
    crop_img = resize_img.crop((left,top,right,bottom))
    np_img = np.array(crop_img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normal_img = (np_img - mean) / std
    #change thd order of dimensions
    normal_img = normal_img.transpose(2,0,1)
    tnr_img = torch.Tensor(normal_img)
    return tnr_img

def Imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax