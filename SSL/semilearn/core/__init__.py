
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


## 
# from .algorithmbase_blood import AlgorithmBase, ImbAlgorithmBase
# from .algorithmbase_celebagender import AlgorithmBase, ImbAlgorithmBase
# from .algorithmbase_celebahair import AlgorithmBase, ImbAlgorithmBase
from .algorithmbase_cifar10 import AlgorithmBase, ImbAlgorithmBase
# from .algorithmbase_cifar100 import AlgorithmBase, ImbAlgorithmBase
# from .algorithmbase_derma import AlgorithmBase, ImbAlgorithmBase
# from .algorithmbase_eurosat import AlgorithmBase, ImbAlgorithmBase
# from .algorithmbase_FFHQ import AlgorithmBase, ImbAlgorithmBase
# from .algorithmbase_path import AlgorithmBase, ImbAlgorithmBase
# from .algorithmbase_SVHN import AlgorithmBase, ImbAlgorithmBase


from .utils.registry import import_all_modules_for_register

import_all_modules_for_register()