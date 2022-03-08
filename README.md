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
