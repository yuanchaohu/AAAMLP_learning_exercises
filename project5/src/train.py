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

from torchvision.models import resnet18, ResNet18_Weights
from sklearn import metrics
from sklearn.model_selection import train_test_split

from wtfml.data_loaders import image
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
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.based_model = torchvision.models.resnet18(pretrained=True)
        self.based_model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False).eval()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str
    )
    parser.add_argument(
        "--device",
        type=str
    )
    parser.add_argument(
        "--epochs",
        type=int
    )
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    images = df.image_id.values.tolist()
    images = [
        os.path.join(args.data_path, "images", i+".png")
        for i in images
    ]
    targets = df[["healthy", "multiple_diseases", "rust", "scab"]].values

    model = Model()
    model.to(args.device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean,
                std,
                max_pixel_value=255.0,
                always_apply=True
            )
        ]
    )

    train_images, valid_images, train_targets, valid_targets = train_test_split(images, targets)

    train_loader = ClassificationDataLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=(128, 128),
        augmentations=aug
    ).fetch(
        batch_size=16,
        num_workers=4,
        drop_last=False,
        shuffle=True,
        tpu=False
    )

    valid_loader = ClassificationDataLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(128, 128),
        augmentations=aug
    ).fetch(
        batch_size=16,
        num_workers=4,
        drop_last=False,
        shuffle=False,
        tpu=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.6
    )

    for epoch in range(args.epochs):
        train_loss = Engine.train(
            train_loader, model, optimizer
        )
        valid_loss = Engine.evaluate(
            valid_loader, model
        )
        print(
            f"{epoch}, Train Loss={train_loss}, Valid Loss={valid_loss}"
        )