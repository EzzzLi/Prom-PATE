from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
import timm

class ConvNeXTv2_base_224_22k(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.pre_model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-22k-224")
        self.nnlayer = nn.Sequential(nn.Linear(1000, num_classes))
        self.trans = transforms.Resize(224)
        
    def forward(self, image):
        image = self.trans(image)
        x = self.pre_model(image,output_hidden_states=True)
        # x = x.to_tuple()
        # print(type(x))
        output = self.nnlayer(x['logits'])
        feat = x['hidden_states']
        # print(feat[-1])
        return {'logits':output, 'feat':feat[-1]}

def convnextv2_base_224_22k(pretrained=False, pretrained_path=None, **kwargs):
    model = ConvNeXTv2_base_224_22k(**kwargs)
    return model

class ConvNeXTv2_large_224_22k(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.pre_model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-large-22k-224")
        ## test
        for param in self.pre_model.parameters():
            param.requires_grad = False
        self.nnlayer = nn.Sequential(nn.Linear(1000, num_classes))
        self.trans = transforms.Resize(224)
        
    def forward(self, image):
        image = self.trans(image)
        x = self.pre_model(image,output_hidden_states=True)
        # x = x.to_tuple()
        # print(type(x))
        output = self.nnlayer(x['logits'])
        feat = x['hidden_states']
        # print(feat[-1])
        return {'logits':output, 'feat':feat[-1]}

def convnextv2_large_224_22k(pretrained=False, pretrained_path=None, **kwargs):
    model = ConvNeXTv2_large_224_22k(**kwargs)
    return model

class ConvNeXTv2_large_ft_in22k_in1k(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.pre_model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', pretrained=True)
        self.pre_model = self.pre_model.eval()
        self.nnlayer = nn.Sequential(nn.Linear(1000, num_classes))
        self.trans = transforms.Resize(224)
        
    def forward(self, image):
        image = self.trans(image)
        x = self.pre_model(image)
        # print("************************")
        # print(x)
        output = self.nnlayer(x)
        # print(feat[-1])
        return {'logits':output, 'feat':output}
    
def convnextv2_large_ft_in22k_in1k(pretrained=False, pretrained_path=None, **kwargs):
    model = ConvNeXTv2_large_ft_in22k_in1k(**kwargs)
    return model
