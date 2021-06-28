'''
http server is a N:M ReID
'''

from flask import Flask, jsonify, request, redirect, render_template

import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import math
from models.resnet50 import Net
from PIL import ImageFile
import torch.backends.cudnn as cudnn
from base64_func import *


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


def get_featureV2(model, img_list1: list, transforms, ms):
    img_all = torch.FloatTensor()
    for data in img_list1:
        img = transforms(data)
        img = torch.unsqueeze(img, 0)
        img_all = torch.cat((img_all, img), 0)
    features = torch.FloatTensor()
    print('-' * 10)
    print("Input img shape:", img_all.shape)
    n, c, h, w = img_all.size()
    print(type(img_all))

    ff = torch.FloatTensor(n, 512).zero_()
    if torch.cuda.is_available():
        ff = torch.FloatTensor(n, 512).zero_().cuda()
    # print("ff:", ff.shape)
    for i in range(2):
        if (i == 1):
            img_all = fliplr(img_all)

        input_img = Variable(img_all)
        if torch.cuda.is_available():
            # print("torch.cuda.is_available():", torch.cuda.is_available())
            input_img = Variable(img_all.cuda())

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
    print("Output features shape:", features.shape)
    return features


print("加载模型")
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("use gpu:", use_gpu)
    torch.cuda.set_device("cuda:0")
    cudnn.benchmark = True

data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_structure = Net(751, stride=2)
assert os.path.exists("./model_weight/net_59.pth") == True, "权重文件不存在"
model = load_network(model_structure, network_weights_path="./model_weight/net_59.pth")

model.classifier.classifier = nn.Sequential()

model = model.eval()
if use_gpu:
    model = model.cuda()
#
ms = "1"
print('We use the scale: %s' % ms)
str_ms = ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

print("加载完成")

allowed_extension = ['png', 'jpg', 'jpeg']

app = Flask(__name__)


@app.route('/get_feature', methods=['POST'])
def get_feature():
    params = request.json if request.method == "POST" else request.args
    images = base64_to_image(params["image_list"])
    features = get_featureV2(model, images, data_transforms, ms)
    #touch to list
    features_list=[]
    for item in features:
        features_list.append(item.tolist())
    result={
        'code':200,
        'message':"Success",
        'data':features_list
    }
    return jsonify(result)

@app.route('/', methods=['POST'])
def reid_run():
    params = request.json if request.method == "POST" else request.args
    print(type(params["img1_list"]))
    # print(params["img2_list"])

    t = params["t"]

    img1_PIL = base64_to_image(params["img1_list"])
    # base64_save(params["img1_list"])
    img2_PIL = base64_to_image(params["img2_list"])
    if True:
        features1 = get_featureV2(model, img1_PIL, data_transforms, ms)
        features1 = np.around(features1, 4)

        features2 = get_featureV2(model, img2_PIL, data_transforms, ms)
        features2_T = np.around(np.transpose(features2, [1, 0]), 4)

        with torch.no_grad():
            # 矩阵乘法torch.mm [1,2]x[2,3]=[1,3]
            score = np.dot(features1, features2_T)
            print("score:", score)
        pairs1 = []
        pairs2 = []
        scores = []
        for i, s in enumerate(score):
            max_index = np.argmax(s)

            if float(s[max_index]) < float(t):
                pairs1.append(i)
                pairs2.append(-1)
                scores.append(0)
                continue

            scores.append(int(s[max_index]))
            pairs1.append(i)
            pairs2.append(int(max_index))
    # 返回结果
    return get_result(200, "Success", scores, {"cam1": pairs1, "cam2": pairs2})


def softmax(x: list) -> list:
    ret = []
    x = np.exp(x)
    for i, data in enumerate(x):
        e_data = np.exp(data)
        data = e_data / x
        ret.append(data)
    return ret


# 构建接口返回结果
def get_result(code, message, score, data):
    result = {
        "code": code,
        "message": message,
        "score": score,
        "return": data
    }
    print("Response data:", result)
    return jsonify(result)


app.config['JSON_AS_ASCII'] = False
app.run(host='0.0.0.0', port=24415, debug=False, use_reloader=False)