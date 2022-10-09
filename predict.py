import torch
import numpy as np
import argparse
import json
from torchvision import datasets, transforms, models
from PIL import Image

from model import Classifier

parser = argparse.ArgumentParser()

parser.add_argument('image_path',
                    help='Image directory path')
parser.add_argument('checkpoint',
                    help='Checkpoint of the model')
parser.add_argument('--top_k', action='store',
                    dest='topk',
                    default=5,
                    help='Top k prediction probabilities')
parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    default='cat_to_name.json',
                    help='Json association between category and names')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Set training to gpu')


args = parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    process_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    # TODO: Process a PIL image for use in a PyTorch model
    np_image = process_image(pil_image)
    
    return np_image

def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load("model_checkpoint.pth")
    
    model = models.vgg11(pretrained=True)
    model.name = "vgg11"
    
    
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    
    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

model = load_checkpoint(args.checkpoint)

device = ("cuda" if args.gpu else "cpu")
model.to(device)
    
image = process_image(args.image_path).to(device)
# Adding dimension to image (first dimension)
np_image = image.unsqueeze_(0)

model.eval()
with torch.no_grad():
    logps = model.forward(np_image)

ps = torch.exp(logps)
top_k, top_classes_idx = ps.topk(int(args.topk), dim=1)
top_k, top_classes_idx = np.array(top_k.to('cpu')[0]), np.array(top_classes_idx.to('cpu')[0])

# Inverting dictionary
idx_to_class = {x: y for y, x in model.class_to_idx.items()}

top_classes = []
for index in top_classes_idx:
    top_classes.append(idx_to_class[index])
    
if args.category_names != None:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        top_class_names = [cat_to_name[top_class] for top_class in list(top_classes)]
        print(f'Top {args.topk} probabilities: {list(top_k)}')
        print(f'Top {args.topk} classes: {top_class_names}')
else:
    print(f'Top {args.topk} probabilities: {list(top_k)}')
    print(f'Top {args.topk} classes: {list(top_classes)}')