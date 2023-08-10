# Prom-PATE


The official code of ICCV2023 paper "Exploring the Benefits of Visual Prompting in Differential Privacy"


# Usage
## Prepare your environment

This project is running on Python 3.8, but a higher version of Python should also work. Download required packages

```
pip install -r requirements.txt
```

## Train Teacher model

For Cifar10 Dataset

```
cd pate
python pate_cifar10.py
```
For Cifar100 Dataset
```
cd pate
python pate_cifar100.py
```
For Blood_MNIST Dataset
```
cd pate
python pate_blood.py
```

## Get noisy votes

For Cifar10 Dataset

epsilon = 1
```
cd noisycount
python noisy_and_count.py --dataset CIFAR10 \
    --preds_file cifar10 \
    --class_num 10 \
    --seed 8872574 \
    --sigma1 200 \
    --sigma2 50 \
    --threshold 600 \
    --delta 1e-5 \
    --queries 1000 \
    --eps e1 \
```

For Cifar100 Dataset

epsilon = 8
```
cd noisycount
python noisy_and_count.py --dataset CIFAR100 \
    --preds_file cifar100\
    --class_num 100 \
    --seed 8872574 \
    --sigma1 50 \
    --sigma2 10 \
    --threshold 122 \
    --delta 1e-5 \
    --queries 2000 \
    --eps e8
```
epsilon = 4

```
cd noisycount
python noisy_and_count.py --dataset CIFAR100 \
    --preds_file cifar100\
    --class_num 100 \
    --seed 8872574 \
    --sigma1 100 \
    --sigma2 25 \
    --threshold 200 \
    --delta 1e-5 \
    --queries 2000 \
    --eps e4
```

## Train Student model by SSL

For different datasets, first modify the __init__.py file in Prompt-PATE\SSL\semilearn\core.

For Cifar10 Dataset
```
cd SSL
python train.py --c cifar10.yaml
```