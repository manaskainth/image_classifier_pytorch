import argparse
import torch
from torch import nn , optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

def args():
    
    parser = argparse.ArgumentParser(
        description='Provide image_dir, save_dir, arch, learn rate, hidden_units, epochs , device- gpu or cpu ')

    parser.add_argument('--indir', type=str, default='flowers/',
                        help='path to image data')
    parser.add_argument('--out', type=str, default='checkpoints/',
                        help='path for saving checkpoint')
    parser.add_argument('--epoch', type=int,
                        default=3, help='epochs for model')
    parser.add_argument('--device', type=str,
                        default='gpu', help='select cpu or gpu')
    parser.add_argument('--arch', type=str,
                        default='densenet121', help='vgg16 or densenet121')
    parser.add_argument('--learn_rate', type=float,
                        default=0.0003, help='learning_rate for model')
    parser.add_argument('--layers', type=int,
                        default=512, help='hidden_units for model')
    
    return parser.parse_args()


def load(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {'train_data' : transforms.Compose([transforms.Resize(256),
                                                          transforms.CenterCrop(224),
                                                     transforms.RandomRotation(30),
                                                     transforms.RandomVerticalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485,0.456,0.406],
                                                                         [0.229,0.224,0.225]) ]),
                   'test_data' : transforms.Compose([transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485,0.456,0.406],
                                                                         [0.229,0.224,0.225])
                                                     ])
                  }

    # Loading the datasets with ImageFolder
    imageData = {}
    imageData['train_data'] = datasets.ImageFolder(train_dir,transform = data_transforms['train_data'])
    imageData['valid_data'] = datasets.ImageFolder(valid_dir,transform = data_transforms['test_data'])
    imageData['test_data'] = datasets.ImageFolder(test_dir, transform= data_transforms['test_data'])


    # Using the image datasets and the trainforms, define the dataloaders
    dataloader = {}
    dataloader['trainloader'] = torch.utils.data.DataLoader(imageData['train_data'], batch_size=32, shuffle=True)
    dataloader['validloader'] = torch.utils.data.DataLoader(imageData['test_data'],batch_size=32,shuffle=True)
    dataloader['testloader'] = torch.utils.data.DataLoader(imageData['test_data'],batch_size=32,shuffle=True)  
    
    return imageData, dataloader



def create_model(model_name,hidden_layers) :
    
    if model_name=='vgg16':
        model = models.vgg16(pretrained=True)
        features = model.classifier[0].in_features
    else :    
        model = models.densenet121(pretrained=True)
        features = model.classifier.in_features

    # freezing model parameters
    for param in model.parameters():
        param.reqires_grad = False
    
    # Creating the classifier for the required flower classes (102)
    
   
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(features,hidden_layers)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(hidden_layers, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))


    model.classifier = classifier
    return model

def validation(model,data,criterian,device):
    model.eval()
    model.to(device)
    accuracy = 0
    loss = 0

    for images,labels in data:

        images,labels = images.to(device),labels.to(device)
        output = model.forward(images)
        loss+= criterian(output,labels).item()

        ps = torch.exp(output)
        eq = (labels.data == ps.max(dim=1)[1])
        accuracy += eq.type(torch.FloatTensor).mean()

    return accuracy,loss


def trainer(model,train_data,criterion,optimizer,valid_data,epoch,device):
    running_loss = 0

    # Using GPU if available else CPU
    model.to(device)

    for i in range(epoch):
        model.train()
        for loop ,(inputs, labels) in enumerate(train_data):
            loop = loop+1
            inputs,labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            #Forwarding and BackPropagating

            output = model.forward(inputs)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #defining statistics

            if loop % 32 == 0:

            # Evaluation Mode
                model.eval()
                
                with torch.no_grad():
                    accuracy,validation_loss = validation(model,valid_data,criterion,device)
                
                print("Epoch: {}/{} ".format(i+1,epoch),
                     "Running Loss: {:.3f}".format(running_loss/32),
                     "Validation Loss: {:.3f}".format(validation_loss/len(valid_data)),
                     "Validation Accuracy: {:.2f} %".format((accuracy/len(valid_data))*100)  
                     )

                running_loss = 0

                # Training mode
                model.train()
 
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)