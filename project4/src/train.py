import os 
import sys
from sqlalchemy.orm.interfaces import CriteriaOption
import torch

import numpy as np 
import pandas as pd 
import segmentation_models_pytorch as smp 
import torch.nn as nn 
import torch.optim as optim

from apex import activate, amp 
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from dataset import SIIMDataset

TRAINING_CSV = "./train.csv"
TRAINING_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4
EPOCHS = 10
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda"

def train(dataset, data_loader, model, criterion, optimizer):
    """for one epoch"""

    model.train()

    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)
    for d in tk0:
        inputs = d["image"]
        targets = d["mask"]

        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        optimizer.step()
    tk0.close()

def evaluate(dataset, data_loader, model, criterion):
    model.eval()
    final_loss = 0

    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)

    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            output = model(inputs)
            loss = criterion(output, targets)
            final_loss += loss 
    tk0.close()
    return final_loss/num_batches


if __name__ == "__main__":
    df = pd.read_csv(TRAINING_CSV)
    
    df_train, df_valid = model_selection.train_test_split(
        df, random_state=42, test_size=0.1
    )

    training_images = df_train.image_id.values
    validation_images = df_valid.image_id.values

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=None
    )

    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )
    model.to(DEVICE)

    train_dataset = SIIMDataset(
        training_images,
        transform=True,
        preprocessing_fn=prep_fn
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=12
    )

    valid_dataset = SIIMDataset(
        validation_images,
        transform=False,
        preprocessing_fn=prep_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True
    )

    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O1", verbosity=0
    )

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    print(f"training batch size: {TRAINING_BATCH_SIZE}")
    print(f"test batch size: {TEST_BATCH_SIZE}")
    print(f"epochs: {EPOCHS}")
    print(f"image size: {IMAGE_SIZE}")
    print(f"number of training images: {len(train_dataset)}")
    print(f"number of validation images: {len(valid_dataset)}")
    print(f"encoder: {ENCODER}")

    for epoch in range(EPOCHS):
        print(f"training epoch: {epoch}")
        train(
            train_dataset,
            train_loader,
            model,
            criterion,
            optimizer
        )

        print(f"validation epoch: {epoch}")
        val_log = evaluate(
            valid_dataset,
            valid_loader,
            model,
            criterion
        )

        scheduler.step(val_log["loss"])
        print("\n")
