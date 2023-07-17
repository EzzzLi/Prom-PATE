# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .resnet import resnet50
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2
from .vit import vit_base_patch16_224, vit_small_patch16_224, vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96
from .bert import bert_base_cased, bert_base_uncased
from .wave2vecv2 import wave2vecv2_base
from .hubert import hubert_base
from .convnext import convnextv2_base_224_22k, convnextv2_large_224_22k, convnextv2_large_ft_in22k_in1k
from .eva import eva_large_patch14_196_in22k_ft_in22k_in1k
from .swin import swin_base_patch4_window7_224_in22k
