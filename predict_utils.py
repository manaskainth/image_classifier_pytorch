import torch
from torch import nn , optim
from torchvision import transforms, models
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

from train_utils import create_model
import argparse

def args():
    ''' Parses Command Line Arguments
    '''

    parser = argparse.ArgumentParser(
        description='Requires path for the saved model,device to be used(CPU or GPU), path to mapping file, topk probabilities  ')

    parser.add_argument('--img', type = str,
                        default ='flowers/test/19/image_06155.jpg', help = 'path for test image')
    parser.add_argument('--checkpoint', type = str,
                        default ='checkpoints/densenet121.pth', help = 'path for checkpoint')
    parser.add_argument('--device', type = str,
                        default ='gpu', help = 'select cpu or gpu')
    parser.add_argument('--topk', type = int,
                        default ='1', help = 'value for topk predictions')
    parser.add_argument('--map', type = str, default = 'cat_to_name.json',
                         help = 'file to map labels')

    return parser.parse_args()

def load_checkpoint(path):
    ''' Loads a model form a given checkpoint
    '''

    print("Loading Model................\n")
    checkpoint = torch.load(path,map_location=lambda storage, loc: storage)
    model = create_model(checkpoint['arch'],checkpoint['layers'])
    model.load_state_dict(checkpoint["state_dict"])
    model.optimizer = optim.Adam(model.classifier.parameters())
    model.optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint["mapping"]
    model.epoch = checkpoint["epoch"]
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Processing a PIL image for use in a PyTorch model

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])

    pil_img = Image.open(image)

    return transform(pil_img).numpy()

def predict(image_path, model, mapfile, device,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    print("\nClassifying Image.............. \n")
    image = torch.unsqueeze((torch.Tensor(process_image(image_path))),0)
    with open(mapfile, 'r') as f:
        cat_to_name = json.load(f)
    with torch.no_grad():
        model.to(device)
        model.eval()
        image = image.to(device)
        output = model.forward(image)
        result = torch.exp(output.cpu()).topk(topk)
    probs = result[0][0].numpy()
    indices = result[1][0].numpy()
    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    labels = [cat_to_name[x].title() for x in classes]
    return probs,labels,classes,cat_to_name

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
def display(img_path, probs,classes,label_map):


    img_label = img_path.split('/')[-2]
    labels = [label_map[x].title() for x in classes]
    fig, (i_plot, p_plot) = plt.subplots(nrows=2,ncols=1)

    i_plot.axis('off')
    i_plot.set_title(label_map[img_label].title())
    i_plot.imshow(Image.open(img_path))

    ypos = np.arange(len(probs))
    p_plot.barh(ypos,probs)
    p_plot.set_yticks(ypos)
    p_plot.set_yticklabels(labels)
    p_plot.invert_yaxis()
    plt.show()
