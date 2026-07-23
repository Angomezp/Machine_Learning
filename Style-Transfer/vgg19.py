import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from collections import namedtuple

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        vgg_pretrained = vgg19(weights=VGG19_Weights.DEFAULT).features

        # Capas necesarias
        self.slice1 = nn.Sequential(*[vgg_pretrained[i] for i in range(0, 2)])   # relu1_1
        self.slice2 = nn.Sequential(*[vgg_pretrained[i] for i in range(2, 5)])   # relu2_1
        self.slice2b = nn.Sequential(*[vgg_pretrained[i] for i in range(5, 7)])  # relu2_2
        self.slice3 = nn.Sequential(*[vgg_pretrained[i] for i in range(7, 10)])  # relu3_1
        self.slice4 = nn.Sequential(*[vgg_pretrained[i] for i in range(10, 19)]) # relu4_1
        self.slice5 = nn.Sequential(*[vgg_pretrained[i] for i in range(19, 28)]) # relu5_1

        for param in self.parameters():
            param.requires_grad = False

        self.vgg_outputs = namedtuple("VGGOutputs", ['relu1_1', 'relu2_1', 'relu2_2', 'relu3_1', 'relu4_1', 'relu5_1'])

    def forward(self, x):
        h = self.slice1(x)
        relu1_1 = h
        h = self.slice2(h)
        relu2_1 = h
        h = self.slice2b(h)
        relu2_2 = h
        h = self.slice3(h)
        relu3_1 = h
        h = self.slice4(h)
        relu4_1 = h
        h = self.slice5(h)
        relu5_1 = h
        return self.vgg_outputs(relu1_1, relu2_1, relu2_2, relu3_1, relu4_1, relu5_1)
