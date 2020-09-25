import torch
import torch.nn as nn
# from mypath import Path

from torch.nn import functional as F

class resbasicblock(nn.Module):
    expansion=1
    def __init__(self, in_channel, channel, stride=1):
        super(resbasicblock, self).__init__()
        # self.size=size
        self.stride = stride
        # self.num_classes=num_classes
        self.in_channel = in_channel
        self.channel = channel
        self.bn = nn.BatchNorm3d(self.channel)
        self.bn1 = nn.BatchNorm3d(self.channel)
        # self.stride=stride
        self.conv = nn.Conv3d(self.in_channel, self.channel, kernel_size=3,padding=1,stride=self.stride)
        self.conv1 = nn.Conv3d(self.channel, self.channel, kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if self.stride != 1 or self.in_channel != self.channel * 4:
            self.downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, self.channel, kernel_size=1, stride=(self.stride, self.stride, self.stride),
                          bias=False),
                nn.BatchNorm3d(self.channel)
            )

    def forward(self, input):
        shortcut = self.downsample(input)
        # print(shortcut.shape)
        #print(input.shape)
        input = self.conv(input)
        input = self.bn(input)
        input = self.relu(input)
        input = self.conv1(input)
        input = self.bn1(input)
        output = input + shortcut
        output = self.relu(output)
        return output


class resblock(nn.Module):
    expansion=4
    def __init__(self, in_channel, channel, stride=1):
        super(resblock, self).__init__()
        # self.size=size
        self.stride = stride
        # self.num_classes=num_classes
        self.in_channel = in_channel
        self.channel = channel
        self.bn1 = nn.BatchNorm3d(self.channel)
        self.bn2 = nn.BatchNorm3d(self.channel)
        self.bn3 = nn.BatchNorm3d(self.channel * 4)
        # self.stride=stride
        self.conv = nn.Conv3d(self.in_channel, self.channel, kernel_size=(1, 1, 1))
        self.conv1 = nn.Conv3d(self.channel, self.channel, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                              stride=(self.stride, self.stride, self.stride))
        self.conv2 = nn.Conv3d(self.channel, self.channel * 4, kernel_size=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if self.stride != 1 or self.in_channel != self.channel * 4:
            self.downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, self.channel * 4, kernel_size=1, stride=(self.stride, self.stride, self.stride),
                          bias=False),
                nn.BatchNorm3d(self.channel * 4)
            )

    def forward(self, input):
        shortcut = self.downsample(input)
        # print(shortcut.shape)
        input = self.conv(input)
        input = self.bn1(input)
        input = self.relu(input)
        input = self.conv1(input)
        input = self.bn2(input)
        input = self.relu(input)

        input = self.conv2(input)
        input = self.bn3(input)
        # print(output1.shape)
        output = input + shortcut
        output = self.relu(output)
        return output


class Resnet(nn.Module):
    """
    The Cost network.
    """

    def __init__(self, num_classes, block, layers, pretrained=False):
        super(Resnet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.5)
        # self.fc = nn.Linear(512 * 4, num_classes)
        if block == resblock:
            self.fc1 = nn.Linear(2048, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, num_classes)
        else:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, num_classes)
        self.__init_weight()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        # print(out.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        # out = self.fc(out)
        #print
        out = self.relu(self.fc1(out))
        #out = self.dropout(out)
        out = self.relu(self.fc2(out))
        #out = self.dropout(out)

        out = self.fc3(out)
        return out

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride))

        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer4, model.fc1,
         model.fc2]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc3]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    inputs = torch.rand(2, 3, 16, 224, 224)
    #net = Resnet(101, resblock, [3, 4, 6, 3])#resnet50
    #net = Resnet(101, resbasicblock, [2, 2, 2, 2])  # resnet18
    net = Resnet(101, resbasicblock, [3, 4, 6, 3])  # resnet34
    #net = Resnet(101, resblock, [3, 4, 23, 3])  # resnet101
    # net = costblock(64,64,stride=2)
    #print(net.layer4[1].conv.weight.grad)
    outputs = net(inputs)
    print(outputs.size())