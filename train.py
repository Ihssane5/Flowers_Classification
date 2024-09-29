from image_processing import Load_DataSet
import torch.nn as nn
import torch.optim as optim
import torch
import time
import torchvision.models as models
import pandas as pd



def Get_Pretrained_Model(model_name,hidden_units, gpu, num_classes=102,):
    """ Get a Pretrained Model with a  Vgg or Resnet architecture
    Args:
        model_name (str): The  model's architecture
        hidden_units (int): The Number of Neurones in the hidden layer
        num_classes (int): The Number of Predicted Classes
        gpu (bool): using gpu

    Returns:
        model (torch.models) : the Specified Model
    
    """

    if model_name == 'vgg16':
        model = models.vgg16(weights = 'DEFAULT')
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[0].in_features
        model.classifier = nn.Sequential( 
        nn.Linear(n_inputs,hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units,num_classes),
        nn.Dropout(0.2),
        nn.LogSoftmax(dim = 1)
    )
    else:
        model = models.resnet18(weights = 'DEFAULT')
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential( 
        nn.Linear(n_inputs,hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units,num_classes),
        nn.Dropout(0.2),
        nn.LogSoftmax(dim = 1)
    )

    
    return model




def Train_Model(data_loaders, model, lr, n_epochs, gpu,print_every=1,device='cpu'):
    """ Train the choosen model
    Args:
        data_loaders (torch.data.DataLoaders): The data loader
        model (torch.models): The choosed models
        optimizer (torch.nn.optim): The choosed optimizer
        gpu (bool): gpu use

    Returns:
        model (torch.models) : the Specified Model
    
    """
    model.to(device)
    criterion = nn.NLLLoss()
    if isinstance(model, models.VGG):
        optimizer = optim.SGD(model.classifier.parameters(), lr)
    else:
        optimizer = optim.SGD(model.fc.parameters(), lr)


    history = []
    train_loader = data_loaders['train']
    valid_loader = data_loaders['valid']
    test_loader = data_loaders['test']

    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0
        #set the model to training
        model.train()
        #training loop
        for iter, (data,target) in enumerate(train_loader):
            train_start = time.time()
            data, target = data.to(device), target.to(device)
            #clear gradient
            optimizer.zero_grad()
            #prediction are probabilities
            output = model(data) 
            #print(output)
            loss = criterion(output, target)
            #backpropagation of loss
            loss.backward()
            #update the parameters
            optimizer.step()
            #tracking the loss
            train_loss += loss.item()
            #tracking the acurracy
            values, pred = torch.max(output, dim = 1)
            correct_tensor = pred.eq(target)
            accuracy = torch.mean(correct_tensor.type(torch.float16))
            #train accuracy
            train_acc += accuracy.item()
            print(f'Epoch: {epoch}\t {100 * (iter + 1) / len(train_loader):.2f}% complete. {time.time() - train_start:.2f} seconds elpased in iteration {iter + 1}.', end = '\r' )
        #after training loop end start a validation process
        with torch.no_grad():
            model.eval()
            #validation loop
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                #forward pass
                output = model(data)
                #validation loss
                loss = criterion(output, target)
                #tracking the loss
                valid_loss += loss.item()
                #tracking the acurracy
                values, pred = torch.max(torch.exp(output), dim = 1)
                correct_tensor = pred.eq(target)
                accuracy = torch.mean(correct_tensor.type(torch.float16))
                #train accuracy
                valid_acc += accuracy.item()
            #calculate average loss
            train_loss = train_loss / len(train_loader)
            valid_loss = valid_loss / len(valid_loader)
            #calculate average accuracy
            train_acc = train_acc / len(train_loader)
            valid_acc = valid_acc / len(valid_loader)
            history.append([train_loss,valid_loss, train_acc, valid_acc])
            #print training and validation results
            if (epoch + 1 ) % print_every == 0:
                print(f'Epoch: {epoch}\t Training Loss: {train_loss:.4f} \t Validation Loss: {valid_loss:.4f}')
                print(f'Training Accuracy: {100 * train_acc:.4f}%\t Validation Accuracy: {100 * valid_acc:.4f}%')

    checkpoints = {
                    'model_state_dict': model.state_dict(),  # Save model parameters
                    'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
                    'class_to_idx': train_loader.dataset.class_to_idx,# Save any other info you want
                }
    history = pd.DataFrame(history, columns= [
                        'train_loss', 'valid_loss','train_acc','valid_acc'
                    ])
    #testing model
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        #validation loss
        loss = criterion(output, target)
        #tracking the loss
        test_loss += loss.item() * data.size(0) 
        #tracking the acurracy
        values, pred = torch.max(output, dim = 1)
        correct_tensor = pred.eq(target)
        accuracy = torch.mean(correct_tensor.type(torch.float16))
        #train accuracy
        test_acc += accuracy.item() * data.size(0)
    #calculate average loss
    test_loss = test_loss / len(test_loader.dataset)
    #calculate average accuracy
    test_acc = test_acc / len(test_loader.dataset)
    print(f'Test Accuracy: {100 * test_acc}%\t Loss Accuracy: {100 * test_loss:.4f}%')
    return model, checkpoints, history

def Save_Checkpoints(checkpoints,save_path):
    """ Save the checkpoints of the trained model
    Args:
        trained_model (torch.models): The trained model
        save_path (string): the path of checkpoints

    Returns:
        None
    
    """
    torch.save(checkpoints,save_path)
def Load_Checkpoints(save_path,gpu,model_name,hidden_units,num_classes=102):
    """ Load Checkpoints and Rebuild the model
    Args:
        save_path (str): the path of saved network
        model_name (string): the name of the rebuild model

    Returns:
        model (torch.models): the loaded_model
    
    """
    checkpoint = torch.load(save_path,map_location=torch.device('cpu'))
    model = Get_Pretrained_Model(model_name,hidden_units, gpu, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model




    










