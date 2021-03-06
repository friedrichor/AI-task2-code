import os
import sys
from pathlib import Path
import torch
from torch import nn

import models
from utils import FocalLoss
import models

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 3
img_size = 224

# 模型
model = models.densenet(num_classes)

# 损失函数
loss_function = FocalLoss(num_classes)

# 模型保存文件夹
path_weights = '/content/drive/MyDrive/weights_task2_classify'

# 预测所需模型权重
weights = ''

# 数据集&分类标签 路径
# path_data = '/content/drive/MyDrive/data_AItask2/Image_classify'
# path_data = '/content/drive/MyDrive/data_AItask2/Label_classify'
path_data = '/content/drive/MyDrive/data_AItask2/Label_classify_crop'
path_train = '/content/drive/MyDrive/data_AItask2/Label_classify_crop_train'  # ROOT / 'train'
path_test = '/content/drive/MyDrive/data_AItask2/Label_classify_crop_val'  # ROOT / 'val'
path_json = ROOT / 'class_indices.json'
