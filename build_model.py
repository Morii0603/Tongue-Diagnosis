import torch
import torchvision
import torch.nn as nn
class ResNet(nn.Module):
    def __init__(self, out_features=1, pretrained=True):
        super().__init__()

      
        if pretrained:
            
            resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        else:
            resnet50 = torchvision.models.resnet50()
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        
        self.linear = nn.Linear(resnet50.fc.in_features+35, out_features)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, non_image_features=None):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = torch.concat((x, non_image_features), dim=1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
def ResNet50(out_features=1, pretrained=True):
    if pretrained:
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    else:
        model = torchvision.models.resnet50()
    fc_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_in_features, out_features),
        nn.Sigmoid()
    )
    return model
def ResNet101(out_features=1, pretrained=True):
    if pretrained:
        model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
    else:
        model = torchvision.models.resnet101()
    fc_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_in_features, out_features),
        nn.Sigmoid()
    )
    return model
def Swin_T(out_features=1, pretrained=True):
    if pretrained:
        model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
    else:
        model = torchvision.models.swin_t()
    model.head = nn.Sequential(
        nn.Linear(768, out_features=out_features),
        nn.Sigmoid()
    )
    return model
def efficientnet_v2_s(out_features=1, pretrained=True):
    if pretrained:
        net = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
    else:
        net = torchvision.models.efficientnet_v2_s()
    net.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(1280, 1),
        nn.Sigmoid()
    )
    return net
