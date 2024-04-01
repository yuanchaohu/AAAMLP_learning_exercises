from math import inf, log
import os 
import pandas as pd 
import numpy as np 

import albumentations
import argparse
import torch
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

from sklearn import metrics
from sklearn.model_selection import train_test_split

from wtfml.engine import Engine 
from wtfml.data_loaders.image import ClassificationDataLoader


class DenseCrossEntropy(nn.Module):
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()
    
    def forward(self, logits, labels):
        logits = logits.float()
        labels = logits.float()

        logprobs = F.log_softmax(logits, dim=-1)

        loss = -labels * logprobs
        loss = loss.sum(-1)
        return loss.mean()
    
    def Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.based_model = torchvision.models.resnet18(pretrained=True)
            in_features = self.based_model.fc.in_features
            self.out = nn.Linear(in_features, 4)
        
        def forward(self, image, targets=None):
            batch_size, C, H, W = image.shape
            x = self.based_model.conv1(image)
            x = self.based_model.bn1(x)
            x = self.based_model.relu(x)
            x = self.based_model.maxpool(x)

            x = self.based_model.layer1()
            x = self.based_model.layer2()
            x = self.based_model.layer3()
            x = self.based_model.layer4()

            x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
            x = self.out(x)

            loss = None
            if targets is not None:
                loss = DenseCrossEntropy()(x, targets.type_as(x))
            
            return x, loss 