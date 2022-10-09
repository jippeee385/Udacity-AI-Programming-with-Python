import argparse
import torch
from collections import OrderedDict
from os.path import isdir
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from model import Classifier



# argparser

parser = argparse.ArgumentParser(description="Train.py")
parser.add_argument('--dir', dest="dir", action="store", default="flowers", type = str)
parser.add_argument('--arch', dest="arch", action="store", default="vgg11", type = str)
parser.add_argument('--hidden_units', dest="hidden_units", action="store", default=512, type = int)
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
args = parser.parse_args()

# create the data loading functions
def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data



def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader


# Set directory for training
data_dir = args.dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Pass transforms in, then create trainloader
train_data = test_transformer(train_dir)
valid_data = train_transformer(valid_dir)
test_data = train_transformer(test_dir)

trainloader = data_loader(train_data)
validloader = data_loader(valid_data, train=False)
testloader = data_loader(test_data, train=False)

# set the gpu / cpu
if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'



# TODO: Define your network architecture here


    
# load in the model
model = eval("models." + args.arch +"(pretrained=True)")

# don't compute gradients
for param in model.parameters():
    param.requires_grad = False
    
# set the classifier we want to train    
model.classifier = Classifier()
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)


# Training the model
epochs = args.epochs
steps = 0
running_loss = 0

training_losses, validation_losses = [], []

for e in range(epochs):
    for inputs, targets in trainloader:
        steps += 1
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)                               
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                
                valid_loss += batch_loss.item()
                
                # Accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {e+1}/{epochs} (steps: {steps}).. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(trainloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(trainloader):.3f}.. ")
        
        training_losses.append(running_loss)
        validation_losses.append(valid_loss)
        running_loss = 0
        steps = 0
        model.train()
        
model.class_to_idx = train_data.class_to_idx      
        
checkpoint = {'epochs': epochs,
              'classifier':model.classifier,
              'class_to_idx':model.class_to_idx,
              'training_losses': training_losses,
              'validation_losses': validation_losses,
              'layers': args.hidden_units,
              'optimizer_state_dict': optimizer.state_dict(), 
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, 'model_checkpoint.pth')