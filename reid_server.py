'''
http server is a N:M ReID
'''
import argparse
import math
import os

from flask import Flask, jsonify, request
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from models.resnet50 import Net
import torch.backends.cudnn as cudnn
import numpy
import PIL

# from base64_func import *
from utils import load_network, fliplr,base64_to_image

parser = argparse.ArgumentParser()
parser.add_argument('--weight', '-w', type=str, default='./weight/reid_weight.pth',
                    help='net weigth')
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

print("加载模型")
if CUDA_AVAILABLE:
    torch.cuda.set_device("cuda:0")
    cudnn.benchmark = True

data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_structure = Net(751, stride=2)
assert os.path.exists(args.weight) is True, "权重文件不存在"
model = load_network(model_structure, network_weights_path=args.weight)

model.classifier.classifier = nn.Sequential()

model = model.eval()
model = model.cuda() if CUDA_AVAILABLE else ...
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


def _get_feature(model, imgs: list, transforms, ms):
    img_t = torch.FloatTensor()
    for data in imgs:
        img = transforms(data)
        img = torch.unsqueeze(img, 0)
        img_t = torch.cat((img_t, img), 0)
    features = torch.FloatTensor()

    n, c, h, w = img_t.size()

    ff = torch.FloatTensor(n, 512).zero_()
    ff = ff.cuda() if CUDA_AVAILABLE else ...

    for i in range(2):

        img_t = fliplr(img_t) if i == 1 else ...

        img_t = img_t.cuda() if CUDA_AVAILABLE else img_t
        input_img = Variable(img_t)

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
    print("Output features shape:", features.shape)
    return features


@app.route('/get_feature', methods=['POST'])
def get_feature():
    params = request.json if request.method == "POST" else request.args

    images = base64_to_image(params["image_list"])
    if not images:
        result = {
            'code': 200,
            'message': "Success",
            'data': []
        }
        return jsonify(result)

    features = _get_feature(model, images, data_transforms, ms)
    # touch to list
    features_list = []
    for item in features:
        features_list.append(item.tolist())
    result = {
        'code': 200,
        'message': "Success",
        'data': features_list
    }
    return jsonify(result)


app.config['JSON_AS_ASCII'] = False
app.run(host='0.0.0.0', port=24415, debug=False, use_reloader=False)
