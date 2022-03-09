# Image Segmentation Model

## Usage
```
from model import *
from trainer import *
from metrics import *
import torch

model = AttU_Net(img_ch=3, output_ch=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
measures = {'dice_coef':dice_coef_metric,
           'iou':iou,
           'precision':precision,
           'recall':recall,
           'fscore':fscore}

train_log, val_log = train_model("R2UNet", model, dataloader, bce_dice_loss, optimizer, scheduler, measures, num_epochs)
```

## Dataset
[ISIC 2018 Challenge](https://challenge2018.isic-archive.com/task1/training/) Task 1: Lesion Boundary Segmentation

The dataset (training and validation) was split into three subsets, training set, validation set, and test set (included original validation set), which the proportion is 80%, 10% and 10% of the whole dataset, respectively. The entire dataset contains 2694 images where 2075 images were used for training, 259 for validation and 360 for testing models.

## Result

In the combined image:
- **Purple**: True Positive (model prediction matches lesion area marked by human)
- **Blue**: False Negative (model missed part of lesion in the area)
- **Green**: False Positive (model incorrectly predicted lesion in an area where there is none)

![](./results/attn_1.png)

![](./results/attn_2.png)

![](./results/attn_3.png)

|   Method  |   Attention UNet  |
| :---: | :---: |
| Dice coefficient | 0.850 |
| IoU | 0.743 |
| Precision | 0.865 |
| Recall | 0.841 |
| F1-Score | 0.850 |