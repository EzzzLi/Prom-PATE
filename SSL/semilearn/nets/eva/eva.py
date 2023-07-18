from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
import timm



class EVA_large_patch14_196_in22k_ft_in22k_in1k(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        # self.pre_model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', checkpoint_path = '/root/Semi/model.safetensors')
        self.pre_model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True)
        self.pre_model = self.pre_model.eval()
        self.nnlayer = nn.Sequential(nn.Linear(1000, num_classes))
        self.trans = transforms.Resize(196)
        
    def forward(self, image):
        image = self.trans(image)
        x = self.pre_model(image)
        # print("************************")
        # print(x)
        output = self.nnlayer(x)
        # print(feat[-1])
        return {'logits':output, 'feat':output}
    
def eva_large_patch14_196_in22k_ft_in22k_in1k(pretrained=False, pretrained_path=None, **kwargs):
    model = EVA_large_patch14_196_in22k_ft_in22k_in1k(**kwargs)
    return model