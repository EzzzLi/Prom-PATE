gennoisylabel="python noisy_and_count.py"

## teacher num 
## --dataset:
## --eps: target epsilon, Only used to mark filename
## --seed: random seed, keep the seed same with the pate file
## --preds_file: teachers' vote results
## --sigma1:
## --sigma2:
## --threshold:
## --delta:
## --queries:

## new 100 192 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --eps e1 \
#     --class_num 10 \
#     --seed 8872574 \
#     --preds_file swin_100_192 \
#     --sigma1 150 \
#     --sigma2 50 \
#     --threshold 430 \
#     --delta 1e-5 \
#     --queries 1000 

## new 250 192 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --eps e1 \
#     --class_num 10 \
#     --seed 8872574 \
#     --preds_file swin_250_192 \
#     --sigma1 150 \
#     --sigma2 100 \
#     --threshold 500 \
#     --delta 1e-5 \
#     --queries 1000 

## new 500 192 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --eps e1 \
#     --class_num 10 \
#     --seed 8872574 \
#     --preds_file swin_500_192 \
#     --sigma1 150 \
#     --sigma2 100 \
#     --threshold 650 \
#     --delta 1e-5 \
#     --queries 1000 

### new 1000 192 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 600 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 \

### new 1500 192 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1500_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 55 \
#     --threshold 700 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 \


### new 1500 192 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_2000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 60 \
#     --threshold 720 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 \


#### e = 1,2,4,8
### new 1000 192 e2
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 20 \
#     --threshold 590 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2 \

# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 150 \
#     --sigma2 15 \
#     --threshold 675 \
#     --delta 1e-5 \
#     --queries 1500 \
#     --eps e15 \

# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 150 \
#     --sigma2 20 \
#     --threshold 675 \
#     --delta 1e-5 \
#     --queries 1500 \
#     --eps e15 \


### new 1000 192 e3
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 20 \
#     --threshold 350 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e3 \


## size

## new 1000 64 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_64 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 650 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 \

## new 1000 128 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_128 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 610 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 \


## new 1000 160 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_160 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 610 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 \

## new 1000 210 e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_210 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 570 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 

## mask e1
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_192_nomask \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 600 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 

## nofc ********************************************
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_192_nofc \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 650 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 

## 2fc
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_1000_192_2fc \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 670 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 


## model

## vit
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file vit_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 600 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 

## resnet50
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file resnet50_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 650 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 

## resnet152
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file resnet152_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 620 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 

## wideresnet
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file wideresnet_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 620 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 


## EuroSAT e10
# $gennoisylabel \
#     --dataset EuroSAT \
#     --preds_file swin_eurosat_250_160 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 5 \
#     --threshold 115 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e10


# EuroSAT e3
# $gennoisylabel \
#     --dataset EuroSAT \
#     --preds_file swin_eurosat_250_160 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 10 \
#     --threshold 250 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e3test




# transfer EuroSAT e3
# $gennoisylabel \
#     --dataset EuroSAT \
#     --preds_file eurosat_160_transfer \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 10 \
#     --threshold 250 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e3

# $gennoisylabel \
#     --dataset EuroSAT \
#     --preds_file eurosat_transfer_swin_250 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 10 \
#     --threshold 246 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e3


## EuroSAT e1
# $gennoisylabel \
#     --dataset EuroSAT \
#     --preds_file swin_eurosat_250_160 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 100 \
#     --threshold 410 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 


# blood 1000 192 e3
# $gennoisylabel \
#     --dataset BloodMNIST \
#     --preds_file wideresnet_blood_1000_192 \
#     --class_num 8 \
#     --seed 8872574 \
#     --sigma1 150 \
#     --sigma2 20 \
#     --threshold 360 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e3



## blood 1000 192 e19
# $gennoisylabel \
#     --dataset BloodMNIST \
#     --preds_file wideresnet_blood_1000_192 \
#     --class_num 8 \
#     --seed 8872574 \
#     --sigma1 150 \
#     --sigma2 20 \
#     --threshold 480 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e19

##blood transfer wideresnet
# $gennoisylabel \
#     --dataset BloodMNIST \
#     --preds_file blood_transfer_wideresnet_192 \
#     --class_num 8 \
#     --seed 8872574 \
#     --sigma1 150 \
#     --sigma2 20 \
#     --threshold 490 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e19

##blood transfer wideresnet fully
# $gennoisylabel \
#     --dataset BloodMNIST \
#     --preds_file blood_transfer_wideresnet_192_fully \
#     --class_num 8 \
#     --seed 8872574 \
#     --sigma1 150 \
#     --sigma2 20 \
#     --threshold 470 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e19

##blood transfer wideresnet p47k
# $gennoisylabel \
#     --dataset BloodMNIST \
#     --preds_file blood_1000_wideresnet_transfer_p47k \
#     --class_num 8 \
#     --seed 8872574 \
#     --sigma1 150 \
#     --sigma2 20 \
#     --threshold 500 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e19

#transfer learning 
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file resnet50_transfer_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 670 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 

## fully
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file resnet50_transfer_1000_192_fully \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 670 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 

#transfer learning swin
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_transfer_1000_192_fully \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 630 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 

#transfer learning swin
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file swin_transfer_1000_192 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 620 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 


# transfer learning 
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file resnet_transfer_1000_p50k \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 670 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1

## PATE
# $gennoisylabel \
#     --dataset CIFAR10 \
#     --preds_file PATE_resnet18_1000 \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 50 \
#     --threshold 660 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1 


##cifar100
# teacher_preds_cifar100_swin_250_192
# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file cifar100_swin_250_192 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 10 \
#     --threshold 265 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e4

# teacher_preds_cifar100_swin_250_192
# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file cifar100_swin_250_192 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 10 \
#     --threshold 190 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8


# teacher_preds_cifar100_swin_250_192
# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file cifar100_swin_250_192 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 10 \
#     --threshold 300 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e3

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file cifar100_swin_100_192 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 20 \
#     --threshold 200 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2

## sam swin 
# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file vit_21k_250_192_SAM \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 210 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e4

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file vit_21k_250_192_SAM \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 128 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file vit_21k_250_192_SAM \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 128 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# swin 22k
# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swin_22k_250_192_SAM \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 128 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_22k_200_168 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 115 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_22k_150_168 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 105 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_22k_250_168 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 128 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_22k_300_168 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 137 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_22k_350_168 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 145 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_22k_350_168 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 20 \
#     --threshold 230 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e4

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file convnextv2_large_22k_300_192\
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 137 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file convnextv2_large_22k_300_192\
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 20 \
#     --threshold 220 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e4

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file convnextv2_large_22k_300_192\
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 40 \
#     --threshold 460 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file convnextv2_large_22k_350_192\
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 146 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file convnextv2_large_22k_400_192\
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 155 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8


# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file eva_300_168\
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 115 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8
## eva
# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file eva_350_168\
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 122 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8


# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file eva_350_168\
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 20 \
#     --threshold 200 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e4


# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file eva_350_168\
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 20 \
#     --threshold 450 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2



# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_22k_400_168 \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 153 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# swinv2 22k 192 
# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_192_22k_250_160_SAM \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 128 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_192_22k_250_160_SAM \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 15 \
#     --threshold 220 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e4

# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_192_22k_250_160_SAM \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 200 \
#     --sigma2 10 \
#     --threshold 440 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2


# $gennoisylabel \
#     --dataset CIFAR100 \
#     --preds_file swinv2_192_22k_200_160_SAM \
#     --class_num 100 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 120 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e8
### celeba_gender

# # prom
# $gennoisylabel \
#     --dataset CelebA \
#     --preds_file celeba_1000_swin \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 900 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test


# # Transfer
# $gennoisylabel \
#     --dataset CelebA \
#     --preds_file celeba_1000_swin_transfer \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 900 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# # prom
# $gennoisylabel \
#     --dataset CelebA \
#     --preds_file celeba_1000_vit \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 900 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test


# # Transfer
# $gennoisylabel \
#     --dataset CelebA \
#     --preds_file celeba_1000_vit_transfer \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 900 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# $gennoisylabel \
#     --dataset CelebA \
#     --preds_file celeba_2000_swin \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 500 \
#     --threshold 1800 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test


# # # Transfer
# $gennoisylabel \
#     --dataset CelebA \
#     --preds_file celeba_2000_swin_transfer \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 500 \
#     --threshold 1800 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test



### SVHN

## prom
# $gennoisylabel \
#     --dataset SVHN \
#     --preds_file svhn_500_wideresnet \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 10 \
#     --threshold 415 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2

# $gennoisylabel \
#     --dataset SVHN \
#     --preds_file svhn_250_wideresnet \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 244 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2test



## Transfer
# $gennoisylabel \
#     --dataset SVHN \
#     --preds_file svhn_500_wideresnet_transfer \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 10 \
#     --threshold 410 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2

# $gennoisylabel \
#     --dataset SVHN \
#     --preds_file svhn_250_wideresnet_tansfer \
#     --class_num 10 \
#     --seed 8872574 \
#     --sigma1 50 \
#     --sigma2 10 \
#     --threshold 230 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2test


# tissuemnist
# $gennoisylabel \
#     --dataset TissueMNIST \
#     --preds_file tissue_1000_wideresnet_160 \
#     --class_num 8 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 650 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2


# $gennoisylabel \
#     --dataset TissueMNIST \
#     --preds_file tissue_1000_wideresnet_160 \
#     --class_num 8 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 850 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1


# $gennoisylabel \
#     --dataset TissueMNIST \
#     --preds_file tissue_1000_wideresnet_192_tansfer \
#     --class_num 8 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 635 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps e2

# $gennoisylabel \
#     --dataset TissueMNIST \
#     --preds_file tissue_1000_wideresnet_192_tansfer \
#     --class_num 8 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 800 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1



## pathmnist
# $gennoisylabel \
#     --dataset PathMNIST \
#     --preds_file path_1000_wideresnet_128 \
#     --class_num 9 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 1030 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1

# $gennoisylabel \
#     --dataset PathMNIST \
#     --preds_file path_1000_wideresnet_transfer \
#     --class_num 9 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 1050 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e1



## dermamnist

# $gennoisylabel \
#     --dataset DermaMNIST \
#     --preds_file derma_250_wideresnet_192 \
#     --class_num 7 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 280 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e2


# $gennoisylabel \
#     --dataset DermaMNIST \
#     --preds_file derma_250_wideresnet_192 \
#     --class_num 7 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 140 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test


# $gennoisylabel \
#     --dataset DermaMNIST \
#     --preds_file derma_500_wideresnet_192 \
#     --class_num 7 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 300 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test


# $gennoisylabel \
#     --dataset DermaMNIST \
#     --preds_file derma_250_wideresnet_transfer \
#     --class_num 7 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 280 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps e2

# $gennoisylabel \
#     --dataset DermaMNIST \
#     --preds_file derma_250_wideresnet_transfer \
#     --class_num 7 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 50 \
#     --threshold 140 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# $gennoisylabel \
#     --dataset DermaMNIST \
#     --preds_file derma_500_wideresnet_transfer \
#     --class_num 7 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 300 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

## ffhq
# $gennoisylabel \
#     --dataset FFHQ \
#     --preds_file ffhq_1000_vit \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 800 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# $gennoisylabel \
#     --dataset FFHQ \
#     --preds_file ffhq_1000_vit_transfer \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 800 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# $gennoisylabel \
#     --dataset FFHQ \
#     --preds_file ffhq_1000_swin \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 800 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# $gennoisylabel \
#     --dataset FFHQ \
#     --preds_file ffhq_1000_swin_transfer \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 800 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# $gennoisylabel \
#     --dataset FFHQ \
#     --preds_file ffha_2000_swin_transfer \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 500 \
#     --threshold 1500 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps test

# $gennoisylabel \
#     --dataset FFHQ \
#     --preds_file ffhq_2000_swin \
#     --class_num 2 \
#     --seed 8872574 \
#     --sigma1 90 \
#     --sigma2 450 \
#     --threshold 1500 \
#     --delta 1e-5 \
#     --queries 2000 \
#     --eps test


## cleeba hair
### CelebA Hair
# $gennoisylabel \
#     --dataset CelebAHair \
#     --preds_file celebahair_1000_vit \
#     --class_num 3 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 800 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# $gennoisylabel \
#     --dataset CelebAHair \
#     --preds_file celebahair_1000_vit_transfer \
#     --class_num 3 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 780 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# $gennoisylabel \
#     --dataset CelebAHair \
#     --preds_file celebahair_1000_swin \
#     --class_num 3 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 780 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test

# $gennoisylabel \
#     --dataset CelebAHair \
#     --preds_file celebahair_1000_swin_transfer \
#     --class_num 3 \
#     --seed 8872574 \
#     --sigma1 100 \
#     --sigma2 200 \
#     --threshold 790 \
#     --delta 1e-5 \
#     --queries 1000 \
#     --eps test