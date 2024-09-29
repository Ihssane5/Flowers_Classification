from image_processing import Process_Image
import torch
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

def Predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img_tnr = Process_Image(image_path)
    input_img = img_tnr.unsqueeze(0)
    output  = model(input_img)
    probs, classes = torch.topk(torch.exp(output),topk,dim=1)
    return probs, classes

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

def Process_Image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
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


def Inference(img_path,model,topk,title,cat_to_name):
    #make predictions
    probs, classes = Predict(img_path,model,topk)
    indexes = classes[0,:]
    labels = list(model.class_to_idx.keys())
    probs_list = probs[0,:]
    true_labels = [labels[idx] for idx in indexes]
    classes_name = [ cat_to_name[label] for label in true_labels]
    probs = [elem.item() for elem in probs_list]
     # Sort probabilities and classes
    sorted_probs = probs[::-1]
    sorted_classes_name = classes_name[::-1]

    # Create the figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    tnr_image = Process_Image(img_path)
    Imshow(tnr_image,axs[0],title)
    

    axs[1].barh(sorted_classes_name, sorted_probs)
    axs[1].set_xlabel('Classes')
    axs[1].set_ylabel('Probabilities')
    axs[1].set_title('Class Probabilities')

    # Show the plot
    plt.tight_layout()
    plt.show()