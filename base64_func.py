import base64
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

encode_list = []
'''
base64 code to Image
'''
def base64_to_image(base64_data_list: list)->list:
    image_list=[]
    for i, data in enumerate(base64_data_list):
        byte_data = base64.b64decode(data)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        image_list.append(img)
    return image_list

def base64_encode(img_path=None, file_path=None) -> list:
    encode_list = []
    if file_path is not None:
        for i, img in enumerate(os.listdir(file_path)):
            img_path = os.path.join(file_path, img)
            assert os.path.exists(img_path), "img is not exists!"
            with open(img_path, 'rb') as f:
                img_data = f.read()
                base64_data = base64.b64encode(img_data)
                base64_data = str(base64_data, 'utf-8')
                encode_list.append(base64_data)

                print(type(base64_data))
                # print("Encode is done!")
    else:
        assert os.path.exists(img_path), "img is not exists!"
        with open(img_path, 'rb') as f:
            img_data = f.read()
            base64_data = base64.b64encode(img_data)
            # print(type(base64_data))
            encode_list.append(str(base64_data,'utf-8'))

            # base64_str = str(base64_data, 'utf-8')
            # print("Encode is done!")

    print("Encode is done!")
    return encode_list

'''
base64 code to opencv image
'''
def base64_decode2cv2(base64_data_list: list) -> list:
    """

    Args:
        base64_data: type:list
        i:index of base65_data list
    Returns:

    """
    cv2_img_data_list = []
    for i, data in enumerate(base64_data_list):
        imgData = base64.b64decode(data)
        nparr = np.fromstring(imgData, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2_img_data_list.append(img_np)
    return cv2_img_data_list


imgs_dir = "/home/lu/people_imgs/1/"
imgs_out_dir = "/home/lu/people_imgs/out"
if __name__ == '__main__':
    base64_list = []
    base64_list = base64_encode(file_path=imgs_dir)
    print(len(base64_list))
    img_list=base64_to_image(base64_list)
    # img_list = base64_decode2cv2(base64_list)
    for i, img in enumerate(img_list):
        # cv2.imwrite("/home/lu/people_imgs/out/{}.jpg".format(i), img)
       img.save("/home/lu/people_imgs/out/{}.jpg".format(i))
    # print("len:",len(base64_list[0]))
    # list1=["a","b"]
    # print(len(list1))
