import torch
from torch import nn
from torchvision import models

import params
from model_convnext import convnext_large


def convnext(out_channel: int = 3, weights: str = '', freeze_layers: bool = False):
    model = convnext_large(num_classes=out_channel).to(params.device)
    if weights != "":
        weights_dict = torch.load(weights, map_location=params.device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    if freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    return model


def inception_v3(out_channel: int = 3):
    model = models.inception_v3(False)
    model.aux_logits = False
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, out_channel)
    return model


def efficientnet(outchannel: int = 3):
    model = models.efficientnet_b3(False)
    in_channel = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_channel, outchannel)
    return model


def densenet(out_channel: int = 3):
    model = models.densenet121(False)
    in_channel = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_channel, out_channel)
    return model


def resnet(out_channel: int = 3):
    model = models.resnet50(False)
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, out_channel)
    return model


def vit(out_channel: int = 3):
    model = models.vit_l_32(False)
    in_channel = model.heads[0].in_features
    model.heads = nn.Sequential(nn.Linear(in_channel, out_channel))
    return model


class MixModel1(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(MixModel1, self).__init__()
        self.model1 = convnext(num_classes)
        self.model2 = inception_v3(num_classes)
        # layers
        self.fc = nn.Linear(in_features=2 * num_classes, out_features=num_classes)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x_stack = torch.cat((x1, x2), 1)
        y = self.fc(x_stack.cuda())
        return y


class MixModel2(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(MixModel2, self).__init__()
        self.model1 = convnext(num_classes)
        self.model2 = inception_v3(num_classes)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out = out1 * out2
        return out


class MixModel3(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(MixModel3, self).__init__()
        self.model1 = convnext(256)
        self.model2 = inception_v3(256)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out = torch.cat((out1, out2), 1).cuda()
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class MixModel4(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(MixModel4, self).__init__()
        self.model1 = convnext(num_classes)
        self.model2 = inception_v3(num_classes)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out = (out1 + out2) / 2
        return out
