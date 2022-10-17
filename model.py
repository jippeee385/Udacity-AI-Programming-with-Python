from torch import nn
from torch import optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        #define hidden layers
        self.fc1 = nn.Linear(25088, args.hidden_units)
        #self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(args.hidden_units, 102)   
        
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.10)

        
    def forward(self, x):
        # make sure input tensor is flattened
        #x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x