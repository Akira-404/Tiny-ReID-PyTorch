# Tiny-ReID-PyTorch 

```
My GitHub:https://github.com/Linfeng-Lee
```

# Dataset
Market-1501  
# Prerequisites

- Python 3.6
- GPU Memory >= 2G
- Pytorch 1.8.1

# Usage

## Clone demo

```
git clone https://github.com/Linfeng-Lee/ReID-PyTorch-Tiny
```


## Prepare Dataset

**change your dataset path in prepare.py**  

```
[line:4] download_path = 'D:\ReId_dataset\Market-1501-v15.09.15'

python3 prepare.py
```

## Train

**Train a model by change your dataset path in train.py**

```
[line:59] dir_path="this is your datast path"

python3 train.py 
```

## Test

```bash
python test.py 
```

## HTTP Server

```python
python server.py
```

url:http://your ip address:24415/reid

![image-20210501191946840](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210501191946840.png)