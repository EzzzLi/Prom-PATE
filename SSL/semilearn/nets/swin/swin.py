import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


from semilearn.nets.utils import load_checkpoint
from transformers import  ViTModel, SwinForImageClassification, AutoModelForImageClassification
from PIL import Image
import requests
from torchvision import transforms


class Swin_base_patch4_window7_224_in22k(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.pre_model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
        self.nnlayer = nn.Sequential(nn.Linear(21841, 1000), nn.Linear(1000, num_classes))
        self.trans = transforms.Resize(224)
#         self.pre_model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
#         self.nnlayer = nn.Sequential(nn.Linear(21841, 1000), nn.Linear(1000, 100))
#         self.trans = transforms.Resize(192)
        
    def forward(self, image):
        image = self.trans(image)
        x = self.pre_model(image,output_hidden_states=True)
        # print(x)
        output = self.nnlayer(x['logits'])
        feat = x['reshaped_hidden_states']
        # print(feat[-1])
        return {'logits':output, 'feat':feat[-1]}

def swin_base_patch4_window7_224_in22k(pretrained=False, pretrained_path=None, **kwargs):
    model = Swin_base_patch4_window7_224_in22k(**kwargs)
    return model