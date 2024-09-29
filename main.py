import argparse
import json
from train import Train_Model
from image_processing import Load_DataSet
from train import Get_Pretrained_Model, Save_Checkpoints, Load_Checkpoints
from predict import Predict, Inference
from torchvision import transforms
import torch.nn as nn
import torch
import torch.optim as optim 


"""Main Script that calls functions for Processing Image,
 Training Model, Save and Load checkpoints and model infrence"""







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'Train Data Directory')
    parser.add_argument('--arch', type = str, help = 'The Model Architecture', choices=['resnet18', 'vgg16'])
    parser.add_argument('--lr', type = float, help = 'The Model Learning Rate')
    parser.add_argument('--hidden', type = int, help = 'The Hidden units number')
    parser.add_argument('--epoch', type = int, help = 'The Number of Epochs to train the model for')
    parser.add_argument('--gpu', type = bool, help = 'Enabling GPU')
    #parser.add_argument('--path', type = str, help = 'Saved model path')
    args = parser.parse_args()

    ##specify the device
    device = torch.device('cuda' if args.gpu else 'cpu')

    ##Specifying the Transformations for the DataSet
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    num_classes = 102

    transformations = {'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(img_mean,img_std),
    ]),
        'valid': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(img_mean,img_std),
    ]),
        'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(img_mean,img_std),
    ]),
    }

    data_loaders = Load_DataSet(args.data_dir,transformations,64,True)

    #train_model
    model = Get_Pretrained_Model(args.arch,args.hidden,args.gpu,num_classes=102)
    trained_model, checkpoints, history = Train_Model(data_loaders,model,args.lr,args.epoch,True,1,device)

    #save the model 
    save_path = args.arch + '.pth'
    Save_Checkpoints(checkpoints,save_path)

    #load the model 
    loaded_model = Load_Checkpoints(save_path,args.gpu,args.arch,args.hidden,num_classes)

    # use the model for inference
    img_pth = input('Image Path:')

    #Predict classes and probs
    probs, classes = Predict(img_pth,loaded_model)
     
    #classes to name files
    with open('./cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)


    # Plot Results
    Inference(img_pth,loaded_model,5,'flower image',cat_to_name)
    
    


    