import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import math
from models.resnet50 import Net
from PIL import Image
from PIL import ImageFile


# 加载网络权重
def load_network(network, network_weights_path):
    network.load_state_dict(torch.load(network_weights_path))
    return network


# 水平翻转
def fliplr(img):
    # function arange will reture int64 val
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# 获取图像特征
def get_feature(model, img_path, transforms, ms):
    img = Image.open(img_path).convert("RGB")
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    img = transforms(img)
    img = torch.unsqueeze(img, 0)
    features = torch.FloatTensor()
    n, c, h, w = img.size()
    ff = torch.FloatTensor(n, 512).zero_()
    for i in range(2):
        if (i == 1):
            img = fliplr(img)

        if torch.cuda.is_available():
            input_img = Variable(img.cuda())
        else:
            input_img = Variable(img)

        for scale in ms:
            if scale != 1:
                # bicubic is only  available in pytorch>= 1.1
                input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic',
                                                      align_corners=False)
            outputs = model(input_img)
            ff += outputs
        # norm feature
        # 对ff求l2范数，压缩维度1
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
    features = torch.cat((features, ff.data.cpu()), 0)
    return features


def img_test(img1_path: str, img2_path: str):
    assert os.path.exists(img1_path) is True,"file_path is not exists"
    assert os.path.exists(img2_path) is True,"file_path is not exists"

    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_structure = Net(751, stride=2)
    model = load_network(model_structure, network_weights_path="model_weight/net_59.pth")

    model.classifier.classifier = nn.Sequential()
    model = model.eval()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

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
        score = score.squeeze(1).cpu()
        score = score.numpy()
        print(score)
    return score


def file_test(file_path: str):
    assert os.path.exists(file_path) is True,"file_path is not exists"
    ms = "1"
    stride = 2
    nclasses = 751

    print('We use the scale: %s' % ms)
    str_ms = ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    use_gpu = torch.cuda.is_available()
    # set gpu ids
    if use_gpu:
        torch.cuda.set_device(0)
        # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
        cudnn.benchmark = True

    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_structure = Net(nclasses, stride=stride)

    model = load_network(model_structure, network_weights_path="model_weight/net_59.pth")

    model.classifier.classifier = nn.Sequential()
    model = model.eval()

    if use_gpu:
        model = model.cuda()

    features = torch.FloatTensor()
    label = []
    # root = "../test_imgs"
    # file_path = os.path.join(root, file_path)
    with torch.no_grad():
        img_paths = os.listdir(file_path)
        print(img_paths)
        for img in img_paths:
            img_path = os.path.join(file_path, img)
            feature = get_feature(model, img_path, data_transforms, ms)
            print("img:{},f:{}".format(img_path, feature.shape))

            features = torch.cat((features, feature.data.cpu()), 0)
            label.append(img)
    return features, label


if __name__ == '__main__':
    # ret=img_test("./test_imgs/gallery/g (4).png","./test_imgs/query/q (2).png")

    gallery_features, gallery_label = file_test("test_imgs/gallery")
    query_features, query_label = file_test("test_imgs/query")

    result = {'query_features': query_features.numpy(),
              'query_label': query_label,
              'gallery_features': gallery_features.numpy(),
              'gallery_label': gallery_label}
    scipy.io.savemat("result.mat",result)
