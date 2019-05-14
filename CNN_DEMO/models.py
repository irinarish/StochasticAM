from __future__ import print_function, division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinMod(nn.Linear):
    '''Linear modules with or without batchnorm, all in one module
    '''
    def __init__(self, n_inputs, n_outputs, bias=False, batchnorm=False):
        super(LinMod, self).__init__(n_inputs, n_outputs, bias=bias)
        if batchnorm:
            self.bn = nn.BatchNorm1d(n_outputs, affine=False)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.batchnorm = batchnorm
        self.bias_flag = bias

    def forward(self, inputs):
        outputs = super(LinMod, self).forward(inputs)
        if hasattr(self, 'bn'):
            outputs = self.bn(outputs)
        return outputs

    def extra_repr(self):
        return '{n_inputs}, {n_outputs}, bias={bias_flag}, batchnorm={batchnorm}'.format(**self.__dict__)


class FFNet(nn.Module):
    '''Feed-forward all-to-all connected network
    '''
    def __init__(self, n_inputs, n_hiddens, n_hidden_layers=2, n_outputs=10, nlin=nn.ReLU, bias=False, batchnorm=False):
        super(FFNet, self).__init__()

        self.features = () # Skip convolutional features

        self.classifier = nn.Sequential(LinMod(n_inputs, n_hiddens, bias=bias, batchnorm=batchnorm), nlin())
        for i in range(n_hidden_layers-1):
            self.classifier.add_module(str(2*i+2), LinMod(n_hiddens, n_hiddens, bias=bias, batchnorm=batchnorm))
            self.classifier.add_module(str(2*i+3), nlin())
        self.classifier.add_module(str(len(self.classifier)), nn.Linear(n_hiddens, n_outputs))

        self.batchnorm = batchnorm
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def test(model, data_loader, criterion=nn.CrossEntropyLoss(), label=''):
    '''Compute model accuracy
    '''
    model.eval()
    device = next(model.parameters()).device

    # Added by Ben 11 Jan for Validation stuff...
    if hasattr(data_loader, 'numSamples'):
      N = data_loader.numSamples
    else:
      N = len(data_loader.dataset)
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            test_loss += criterion(output, target).item()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    accuracy = float(correct)/N
    test_loss /= len(data_loader) # loss function already averages over batch size
    if label:
        print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            label, test_loss, correct, N, 100. * accuracy ))
    return accuracy


class LeNet(nn.Module):
    '''Based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
    '''
    def __init__(self, num_input_channels=3, num_classes=10, window_size=32, bias=True):
        super(LeNet, self).__init__()
        self.bias = bias
        self.window_size = window_size
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 6, 5, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, 5, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * int((int((window_size-4)/2)-4)/2)**2, 120, bias=bias),
            nn.ReLU(),
            nn.Linear(120, 84, bias=bias),
            nn.ReLU(),
            nn.Linear(84, num_classes, bias=bias),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG7(nn.Module):

    def __init__(self, num_input_channels=3, num_classes=10, window_size=32, init_weights=True, bias=True):
        super(VGG7, self).__init__()
        self.bias = bias
        self.window_size = window_size
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 128, kernel_size=3, padding=1, bias=bias), # 128C3
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias), # 128C3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # MP2

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias), # 256C3
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias), # 256C3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # MP2

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=bias), # 512C3
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias), # 512C3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # MP2
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * int(window_size/8)**2, 1024, bias=bias),
            nn.ReLU(),
            #  nn.Dropout(),
            nn.Linear(1024, num_classes, bias=bias),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        '''VGG initialization fromtorchvision/models/vgg.py
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ----------------------------------------------
# ResNet
# ----------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockLin(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockLin, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.has_codes = True

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            *self.layer1,
            *self.layer2,
            *self.layer3,
            *self.layer4,

            self.avgpool
        )

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlockLin(self.inplanes, planes, stride, downsample))
        layers.append(nn.ReLU())
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockLin(self.inplanes, planes))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def resnet18(**kwargs):
    "Constructs a ResNet-18 model."
    return ResNet([2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    "Constructs a ResNet-34 model."
    return ResNet([3, 4, 6, 3], **kwargs)
