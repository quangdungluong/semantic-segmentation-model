import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import CFG
from src.model import AttU_Net, R2AttU_Net, R2U_Net, ResNeXtUNet, U_Net


def create_model(model_name=CFG.brain_mri["model_name"], img_ch=CFG.brain_mri["img_ch"], out_ch=CFG.brain_mri["out_ch"]):
    assert model_name in ["unet", "r2_unet", "attn_unet", "r2attn_unet", "resnext_unet"], "Invalid model name"
    if model_name == "unet":
        model = U_Net(img_ch, out_ch).to(CFG.device)
    elif model_name == "r2_unet":
        model = R2U_Net(img_ch, out_ch).to(CFG.device)
    elif model_name == "attn_unet":
        model = AttU_Net(img_ch, out_ch).to(CFG.device)
    elif model_name == "r2attn_unet":
        model = R2AttU_Net(img_ch, out_ch).to(CFG.device)
    else:
        model = ResNeXtUNet(1).to(CFG.device)
    model.load_state_dict(torch.load(CFG.brain_mri["model_path"], map_location=CFG.device))
    return model


def predict(model, image_path):
    model.eval()
    data_transforms = transforms.Compose([
        transforms.Resize((CFG.brain_mri["image_size"], CFG.brain_mri["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    image = image[None, :]
    pred = model(image.to(CFG.device))
    pred = (np.squeeze(pred) > 0.5).detach().numpy()
    return pred

# model = create_model()
# image_path = "./sample_data/TCGA_CS_4941_19960909_13.tif"
# pred = predict(model, image_path)
# img = Image.fromarray(pred)
# img.save("test.png")
