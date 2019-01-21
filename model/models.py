import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models


def model_1():
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(
                                        nn.Linear(25088,5005),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(5005,5005),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(5005,5005)
                                    )
    return model

class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        self.full_model = models.resnet50(pretrained=False)
        self.full_model.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        self.full_model.fc = nn.Linear(8192,5005)

    def forward(self, x):
        featuremaps = []
        for Id, module in self.full_model._modules.items():
            if Id == 'avgpool':
                break
            x = module(x)        
        x = self.full_model.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.full_model.fc(x)
        return x