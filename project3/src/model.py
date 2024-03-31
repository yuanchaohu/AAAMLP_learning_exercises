from re import L
from sklearn.base import OutlierMixin
import torch.nn as nn 
import pretrainedmodels

def get_model(pretrained, name="alexnet"):
    if name=="alexnet":
        if pretrained:
            model = pretrainedmodels.__dict__["alexnet"](
                pretrained="imagenet"
            )
        else:
            model = pretrainedmodels.__dict__["alexnet"](
                pretrained=None
            )
        
        model.last_linear = nn.Sequential(
            nn.BatchNorm1d(4096),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048, eps=1e-5, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1)
        )
        print(model)
        return model    
    
    elif name=="resnet18":
        if pretrained:
            model = pretrainedmodels.__dict__["resnet18"](
                pretrained="imagenet"
            )
        else:
            model = pretrainedmodels.__dict__["resnet18"](
                pretrained=None
            )
        
        model.last_linear = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=512, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048, eps=1e-5, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1)
        )
    
        print(model)
        return model