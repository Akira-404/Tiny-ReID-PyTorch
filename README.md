# Tiny-ReID-PyTorch 

```
GitHub:https://github.com/Linfeng-Lee
```

# Dataset
Market-1501  
# Prerequisites

- Python 3.6
- GPU Memory >= 2G
- Pytorch 1.8.1

# Usage


## Prepare Dataset

**change your dataset path**

```
download_path = 'D:\ReId_dataset\Market-1501-v15.09.15'
python3 prepare.py
```

## Train

Train a model by

Change your dataset path

```
dir_path="this is your datast path"
python3 train.py 
```

## Test

Use trained model to extract feature by

```bash
python test.py 
```