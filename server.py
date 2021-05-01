from flask import Flask, jsonify, request, redirect, render_template

import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import os
import math
from models.resnet50 import Net
from PIL import Image
from PIL import ImageFile
import torch.backends.cudnn as cudnn
import torch


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


def get_feature(model, img_path, transforms, ms):
    assert os.path.exists(img_path) == True, "图片不存在"
    img = Image.open(img_path).convert("RGB")
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    img = transforms(img)
    img = torch.unsqueeze(img, 0)

    features = torch.FloatTensor()

    n, c, h, w = img.size()
    ff = torch.FloatTensor(n, 512).zero_().cuda()
    for i in range(2):
        if (i == 1):
            img = fliplr(img)

        if torch.cuda.is_available():
            # print("torch.cuda.is_available():", torch.cuda.is_available())
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
assert os.path.exists("model_weight/net_59.pth") == True, "权重文件不存在"
model = load_network(model_structure, network_weights_path="model_weight/net_59.pth")

model.classifier.classifier = nn.Sequential()

model = model.eval()
if use_gpu:
    model = model.cuda()

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


@app.route('/reid', methods=['POST'])
def reid_run():
    # 校验请求参数
    if 'file1' not in request.files or 'file2' not in request.files:
        return get_result(-2, "请求参数错误", {})

    # 获取请求参数
    file1 = request.files['file1']
    file2 = request.files['file2']
    print("Reqest params: {'file1': '%s', 'file2': '%s'}" % (file1.filename, file2.filename))

    # 检查文件扩展名
    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return get_result(-1, "存在格式不正确的文件", {})

    file1.save('img1.jpg')
    file2.save('img2.jpg')
    with torch.no_grad():
        feature1 = get_feature(model, './img1.jpg', data_transforms, ms)
        feature2 = get_feature(model, './img2.jpg', data_transforms, ms)

        feature2_T = feature2.view(-1, 1)
        # 矩阵乘法torch.mm [1,2]x[2,3]=[1,3]
        score = torch.mm(feature1, feature2_T)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        os.remove('./img1.jpg')
        os.remove('./img2.jpg')
    # 返回结果
    return get_result(200, "Success", {"score": str(score[0])})


# 检查文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension


# 构建接口返回结果
def get_result(code, message, data):
    result = {
        "code": code,
        "message": message,
        "data": data
    }
    print("Response data:", result)
    return jsonify(result)


app.config['JSON_AS_ASCII'] = False
app.run(host='0.0.0.0', port=24415, debug=False, use_reloader=False)
