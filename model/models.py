import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
import timm


def get_efficientnet(model_name='efficientnet-b0', num_classes=11, pretrained=True):
    if pretrained:
        net = EfficientNet.from_pretrained(model_name)
    else:
        net = EfficientNet.from_name(model_name)
    in_features = net._fc.in_features
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return net


class EfficientNet_ns(nn.Module):
    def __init__(self, model_arch='tf_efficientnet_b3_ns', n_class=11, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class RANZCRResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output


if __name__ == '__main__':
    # model, image_size = get_efficientnet(model_name='efficientnet-b3', num_classes=5000, pretrained=True), 224
    # model, image_size = RANZCRResNet200D(model_name='resnet101', out_dim=11, pretrained=False), 224
    model, image_size = EfficientNet_ns(model_arch='tf_efficientnet_b3_ns', n_class=11, pretrained=True), 224
    # print(model)
    image_size = 512
    model = model.to(torch.device('cpu'))
    from torchsummary import summary
    # input_s = (3, image_size, image_size)
    input_s = (3, image_size, image_size)
    print(summary(model, input_s, device='cpu'))
    # print(model)
    pass