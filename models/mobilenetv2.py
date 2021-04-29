import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import functools as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__

    print("get calss name:", classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim,
                 class_num,
                 droprate,
                 relu=False,
                 bnorm=True,  # betchnormlization
                 num_bottleneck=512,
                 linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        # 执行weights init function
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)

        # init classifier weights
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class Net(nn.Module):
    def __init__(self, class_num=751, droprate=0.5):
        super(Net, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        self.model = model
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(1280, class_num, droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        x = self.avgpool2d(x)
        x = x.reshape(x.shape[0], -1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = Net(751)
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print(output.shape)
