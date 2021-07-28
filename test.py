import os
import math
import argparse

import PIL.JpegImagePlugin
import numpy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn

from models.resnet50 import Net
from PIL import Image
from PIL import ImageFile

parser = argparse.ArgumentParser()
parser.add_argument('--weight', '-w', type=str, default='./weight/reid_weight.pth',
                    help='net weigth')
parser.add_argument('--test_img1', '-i1', type=str, default=None,
                    help='test img path')
parser.add_argument('--test_img2', '-i2', type=str, default=None,
                    help='test img path')
parser.add_argument('--test_file', '-f', type=str, default=None,
                    help='test file of img path')
parser.add_argument('--gpu', '-gpu', type=bool, default=True,
                    help='use gpu or not')
args = parser.parse_args()

Path = str
CvImage = numpy.ndarray
PILImage = PIL.JpegImagePlugin.JpegImageFile
Tensor = torch.Tensor
NumpyType = numpy.ndarray

CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_AVAILABLE = False if args.gpu else ...

if CUDA_AVAILABLE:
    print('Use GPU')
else:
    print('Use CPU')


# 加载网络权重
def load_network(network, network_weights_path: Path):
    assert os.path.exists(network_weights_path) is True, 'network_weights_path is not exists'
    network.load_state_dict(torch.load(network_weights_path))
    return network


# 水平翻转
def fliplr(img: PILImage)->PILImage:
    # function arange will reture int64 val
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# 获取图像特征
def get_feature(model, img_path: Path, transform, ms: str) -> Tensor:
    img = Image.open(img_path).convert("RGB")
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    img = transform(img)
    img = torch.unsqueeze(img, 0)

    features = torch.FloatTensor()

    n, c, h, w = img.size()

    ff = torch.FloatTensor(n, 512).zero_()
    ff = ff.cuda() if CUDA_AVAILABLE else ...

    for i in range(2):
        img = fliplr(img) if i == 1 else ...

        input_img = img.cuda() if CUDA_AVAILABLE else ...
        input_img = Variable(input_img)

        for scale in ms:
            if scale != 1:
                # bicubic is only  available in pytorch>= 1.1
                input_img = nn.functional.interpolate(input_img,
                                                      scale_factor=scale,
                                                      mode='bicubic',
                                                      align_corners=False)
            outputs = model(input_img)
            ff += outputs
        # norm feature
        # 对ff求l2范数，压缩维度1
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
    features = torch.cat((features, ff.data.cpu()), 0)
    return features


def img_test(img1_path: Path, img2_path: Path) -> NumpyType:
    assert os.path.exists(img1_path) is True, "file_path is not exists"
    assert os.path.exists(img2_path) is True, "file_path is not exists"

    if CUDA_AVAILABLE:
        torch.cuda.set_device("cuda:0")
        cudnn.benchmark = True

    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_structure = Net(751, stride=2)
    model = load_network(model_structure, network_weights_path=args.weight)

    model.classifier.classifier = nn.Sequential()

    model = model.eval()
    model = model.cuda() if CUDA_AVAILABLE else ...

    ms = "1"
    print('We use the scale: %s' % ms)
    str_ms = ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    with torch.no_grad():
        feature1 = get_feature(model, img1_path, data_transforms, ms)
        feature2 = get_feature(model, img2_path, data_transforms, ms)

        feature2_T = feature2.view(-1, 1)
        # 矩阵乘法torch.mm [1,2]x[2,3]=[1,3]
        score = torch.mm(feature1, feature2_T)
        score = score.squeeze(1).cpu().numpy()
        # score = score.numpy()
    return score


def file_test(file_path: Path) -> list:
    assert os.path.exists(file_path) is True, "File_path is not exists"
    assert os.path.isfile(file_path) is True, "File_path is not file"
    ms = "1"
    stride = 2
    nclasses = 751

    print('We use the scale: %s' % ms)
    str_ms = ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # set gpu ids
    if CUDA_AVAILABLE:
        torch.cuda.set_device(0)
        # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
        cudnn.benchmark = True

    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_structure = Net(nclasses, stride=stride)

    model = load_network(model_structure, network_weights_path=args.weight)

    model.classifier.classifier = nn.Sequential()

    model = model.eval()
    model = model.cuda() if CUDA_AVAILABLE else ...

    reslut = []
    with torch.no_grad():
        img_paths = os.listdir(file_path)
        for img in img_paths:
            item = dict()
            img_path = os.path.join(file_path, img)
            feature = get_feature(model, img_path, data_transforms, ms)

            item['img'] = img
            item['feature'] = feature
            reslut.append(item)

    return reslut


if __name__ == '__main__':
    if args.test_img1 is not None and args.test_img2 is not None:
        ret_img = img_test(args.test_img1, args.test_img2)
        print(ret_img)
    if args.test_file is not None:
        ret_file = file_test(args.test_file)
        print(ret_file)
