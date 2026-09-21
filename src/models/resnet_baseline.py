import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_classifier(num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model