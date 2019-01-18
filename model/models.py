import torch
import torch.nn as nn
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
