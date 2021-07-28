import torch
import PIL
import base64
from PIL import Image
from io import BytesIO

Path = str
PILImage = PIL.JpegImagePlugin.JpegImageFile


# 加载网络权重
def load_network(network, network_weights_path: Path):
    network.load_state_dict(torch.load(network_weights_path))
    return network


# 水平翻转
def fliplr(img: PILImage) -> PILImage:
    # function arange will reture int64 val
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def base64_to_image(base64_data: list) -> list:
    image_list = []
    for i, data in enumerate(base64_data):
        byte_data = base64.b64decode(data)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        image_list.append(img)
    return image_list
