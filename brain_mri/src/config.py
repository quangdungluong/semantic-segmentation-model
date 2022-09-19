import torch

class CFG:
    device = torch.device("cpu")

    # Brain MRI Flask app
    brain_mri = {
        "model_path" : "./model_ckpt/attn_unet.pt",
        "model_name" : "attn_unet",
        "img_ch" : 3,
        "out_ch" : 1,
        "image_size" : 256,

    }
