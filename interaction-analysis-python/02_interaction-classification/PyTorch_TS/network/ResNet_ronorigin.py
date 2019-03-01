from __future__ import print_function

import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        self.Conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(out_planes)
        self.Conv2 = nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(self.expansion * out_planes)

        if stride != 1 or in_planes != self.expansion * out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes),
            )

    def forward(self, x):

        residual = self.downsample(x) if hasattr(self, 'downsample') else x

        output = self.Conv1(x)
        output = F.relu(self.BN1(output))

        output = self.Conv2(output)
        output = self.BN2(output)

        output += residual
        output = F.relu(output)

        return output


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(Bottleneck, self).__init__()

        self.Conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.BN1 = nn.BatchNorm2d(out_planes)
        self.Conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(out_planes)
        self.Conv3 = nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=1, stride=1, bias=False)
        self.BN3 = nn.BatchNorm2d(self.expansion * out_planes)

        if stride != 1 or in_planes != self.expansion * out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes),
            )

    def forward(self, x):

        residual = self.downsample(x) if hasattr(self, 'downsample') else x

        output = self.Conv1(x)
        output = F.relu(self.BN1(output))

        output = self.Conv2(output)
        output = F.relu(self.BN2(output))

        output = self.Conv3(output)
        output = self.BN3(output)

        output += residual
        output = F.relu(output)

        return output


class ResNet(nn.Module):
    """
    pre activated version of resnet
    """

    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.Conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.BN1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(8)
        
        # after concatenation, 512 + 512 + 512 = 1536 as the input in fc layer
        # print shape
        
        self.fc1 = nn.Linear(1536 * block.expansion, 256 * block.expansion)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256 * block.expansion, num_classes)
        self.fc3 = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_planes, blocks, stride=1):
        layers = []
        layers.append(block(self.in_planes, out_planes, stride))
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, out_planes))
            self.in_planes = out_planes * block.expansion
        return nn.Sequential(*layers)
    
    # helper
    def forward_4_cat(self,x):
        output = self.Conv1(x)
        output = F.relu(self.BN1(output))

        output = self.layer1(output) # conv2d #2 conv2d + conv2d + maxpool2d
        output = self.layer2(output)
        output = self.layer3(output) #  conv2d #3 conv2d + conv2d + conv2d + maxpool2d
        output = self.layer4(output)

        output = self.avgpool(output)
        output = output.view(x.size(0), -1) #1 x 100000
        
        return output
        
    def forward(self, x1,x2,x3):
        
        # input 2 inputs and concatenate them into one and fully connect them
#         output = self.forward_4_cat(x)
#         print(output.shape)

        out1 = self.forward_4_cat(x1)
        out2 = self.forward_4_cat(x2)
        out3 = self.forward_4_cat(x3)
        
        # concatenate 1 x 512 + 1 x 512 + 1 x 512 = 1 x 1536
        output = torch.cat((out1, out2, out3), 1)
        
        # fc = fully connection
        output = self.fc1(output)
        output = self.drop(output)
        output = self.fc2(output)

        return output
    
    

    def get_feature(self, x1,x2):
        
        return self.forward(x1,x2)


def resnet_18(nc=100):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=nc)
    
    return model


def resnet_34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


def resnet_50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def resnet_101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


def resnet_152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model
