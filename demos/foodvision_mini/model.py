import torch
import torchvision

from torch import nn


def create_effnetb1_model(num_classes:int=3,seed:int=42):
    """
    Creates an EFFicientNetB1 feature extractor model and transforms.
    :param num_classes: number of classes in classifier head.
                        Defaults to 3.
    :param seed: random seed value.
                 Defaults to 42.
    :return: feature extractor model.
         transforms (torchvision.transforms): EffNetB1 image transforms.
    """
    # 1. Setup pretrained EffNetB1 weights
    weigts = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    # 2. Get EffNetB2 transforms
    transforms= weigts.transforms()

    # 3. Setup pretrained model
    model=torchvision.models.efficientnet_b1(weights= "DEFAULT")

    # 4. Freeze the base layers in the model (this will freeze all layers to begin with)
    for param in model.parameters():
        param.requires_grad=False

    # 5. Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier=nn.Sequential(nn.Dropout(p=0.2,inplace=True),
                                   nn.Linear(in_features=1280,out_features=num_classes))
    return model,transforms
