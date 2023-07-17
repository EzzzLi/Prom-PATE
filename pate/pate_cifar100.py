import torch
import torchvision
import argparse
import numpy as np
from torch import nn, optim
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from itertools import accumulate
from torch.utils.data import Subset
import copy
import timm
from sam import SAM
from transformers import ViTImageProcessor, ViTModel, SwinForImageClassification, AutoModelForImageClassification, ConvNextV2ForImageClassification

cifar100_std = (0.2675, 0.2565, 0.2761)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar10_std


## this is reprogramming code
class reProgrammingNetwork(nn.Module):
    def __init__(self,args, input_size=224, patch_H_size=192, patch_W_size=192, channel_out=3, device="cpu") -> None:
        super().__init__()
        self.device = device
        self.channel_out = channel_out
        self.input_size = input_size
        if args.model_name == 'wideresnet':
            self.pre_model = torchvision.models.wide_resnet50_2(pretrained=True)
        elif args.model_name == 'resnet50':
            self.pre_model = torchvision.models.resnet50(pretrained=True)
        elif args.model_name == 'resnet152':
            self.pre_model = torchvision.models.resnet152(pretrained=True)
        elif args.model_name == 'swin':
            self.pre_model = torchvision.models.swin_v2_s(pretrained=True)
        elif args.model_name == 'vit':
            self.pre_model = torchvision.models.vit_b_32(pretrained=True)
        elif args.model_name == 'vit_21k':
            self.pre_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.model_name == 'swin_22k':
            self.pre_model = SwinForImageClassification.from_pretrained("microsoft1/swin-base-patch4-window7-224-in22k/")
        elif args.model_name == 'swinv2_22k':
            self.pre_model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
        elif args.model_name == 'swinv2_22k_ft_1k':
            self.pre_model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft")
        elif args.model_name == 'swinv2_large_22k':
            self.pre_model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
        elif args.model_name == 'convnextv2_ft_in22k_in1k':
            self.pre_model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', pretrained=True)
        elif args.model_name == 'convnextv2_base_22k':
            self.pre_model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-22k-224")
        elif args.model_name == 'convnextv2_large_22k':
            self.pre_model = ConvNextV2ForImageClassification.from_pretrained("convnext/convnextv2-large-22k-224")
        elif args.model_name == 'eva':
            self.pre_model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True)
            # self.pre_model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', checkpoint_path='./eva/model.safetensors')

        self.pre_model.eval()
        for pram in self.pre_model.parameters():
            pram.requires_grad = False
        
        self.M = torch.ones(channel_out, input_size, input_size, requires_grad=False, device=device)
        self.H_start = input_size // 2 - patch_H_size // 2
        self.H_end = self.H_start + patch_H_size        
        self.W_start = input_size // 2 - patch_W_size // 2
        self.W_end = self.W_start + patch_W_size
        self.M[:,self.H_start:self.H_end,self.W_start:self.W_end] = 0
        
        self.W = Parameter(torch.randn(channel_out, input_size, input_size, requires_grad=True, device=device))
        # self.new_layers = nn.Sequential(nn.Linear(768, 100)) ## vit imaganet 21
        # self.new_layers = nn.Sequential(nn.Linear(21841, 1000), nn.Linear(1000, 100)) #swinv2_22k
        self.new_layers = nn.Sequential(nn.Linear(1000, 100))#swinv2_22k_ft_1k and convnextv2_ft_in22k_in1k and convnext_base_22k
        

    def hg(self, imagenet_label):
        return imagenet_label[:,:10]
    
    def forward(self, image):
        X = torch.zeros(image.shape[0], self.channel_out, self.input_size, self.input_size)
        X[:,:,self.H_start:self.H_end,self.W_start:self.W_end] = image.repeat(1,1,1,1).data.clone()
        X = Parameter(X, requires_grad=True).to(self.device)

        P = torch.tanh(self.W * self.M)
        X_adv = P + X
        Y = self.pre_model(X_adv)
        # Y = self.new_layers(Y[1]) ## vit image 21k
        # Y = self.new_layers(Y[0]) ## swin image 22k
        Y = self.new_layers(Y)
        return Y
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_w(model):
    for p in model.parameters():
        if p.requires_grad:
            return p;
def train_model(dataset, test_dataset, args):
    device = args.device
    num_epochs = args.epoch
    batch_size = args.batch_size
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = reProgrammingNetwork(args,input_size=args.target_size, patch_H_size=args.size, patch_W_size=args.size,device=device).to(device)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
    ## SAM optimizer
    base_optimizer = optim.SGD
    optimizer = SAM(filter(lambda p: p.requires_grad, model.parameters()), base_optimizer, lr = args.lr, momentum = 0.9)
    scheduler = StepLR(optimizer, args.LR_step, gamma=args.gamma)
    best_test_acc = 0;
    end_train_acc = 0;
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        test_acc = 0
        for i, (image, label) in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            # print(image.size())
            # processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
            # image = processor(images=image, return_tensors="pt")
            label_hat = model(image)
            loss = loss_function(label_hat, label)
            train_loss += loss.item()
            train_acc += sum(label.cpu().numpy() == label_hat.data.cpu().numpy().argmax(1))
            loss.backward()
            # optimizer.step()
            optimizer.first_step(zero_grad=True)
            
            loss_function(model(image), label).backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)
            
        scheduler.step()
        end_train_acc = train_acc/len(dataset)
        print("Epoch: {}".format(epoch+1),
              "Train Loss: {:.3f}".format(train_loss/len(trainloader)),
              "Train Accuracy: {:.3f}".format(train_acc/len(dataset)),
              "lr: {}".format(scheduler.get_last_lr()[0]))
        # test
        # if epoch == num_epochs-1:
        #     model.eval()
        #     with torch.no_grad():
        #         for image, label in testloader:
        #             image = image.to(device)
        #             preds = model(image).data.cpu().numpy().argmax(1)
        #             test_acc += sum(label.cpu().numpy() == preds)
        #     testacc = test_acc/float(len(test_dataset))
        #     print("Test Accuracy: {:.3f}".format(testacc))
        #     if testacc > best_test_acc:
        #           best_test_acc = testacc
        #     print("Best Test acc:", best_test_acc)
        
    # torch.save(model.state_dict(), checkpoint_file)
    return end_train_acc, best_test_acc, model

def predict_model(model, dataloader, device="cpu"):
    preds_list = []
    model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(dataloader):
            image = image.to(device)
            preds = model(image).data.cpu().numpy().argmax(1)
            preds_list.extend(preds)
    return preds_list



def random_split(dataset, lengths, seed):
    torch.manual_seed(seed)
    indices = torch.randperm(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(accumulate(lengths), lengths)]


def get_dataset(dataset = 'CIFAR10'):
    if dataset == 'CIFAR100':
        train_transform = transforms.Compose([
            transforms.Resize(args.size),             # resize shortest side to 224 pixels
            transforms.CenterCrop(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
            ])
        test_transform = transforms.Compose([
            transforms.Resize(args.size),             # resize shortest side to 224 pixels
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
            ])
        train_dataset = datasets.CIFAR100('../data/CIFAR100', train=True, transform=train_transform, download=True, )
        private_dataset = datasets.CIFAR100('../data/CIFAR100', train=False, transform=test_transform, download=True, )
    return 

def main():
    parser = argparse.ArgumentParser(description='pate train')
    parser.add_argument('--seed', type=int, default=8872574)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--teacher_num', type=int, default=400)
    parser.add_argument('--dataset', type=str, default='CIFAR100')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--LR_step', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--size', type=int, default=168)
    parser.add_argument('--target_size', type=int, default=196)
    parser.add_argument('--model_name', type=str, default='eva',
                       choices=['wideresnet', 'resnet50', 'resnet152', 'swin', 'vit', 'vit_21k', 'swin_22k', 'swinv2_22k', 'swinv2_22k_ft_1k', 'swinv2_large_22k', 'convnextv2_ft_in22k_in1k', 'convnextv2_base_22k', 'convnextv2_large_22k', 'eva'])
    parser.add_argument('--num_classes', type=int, default=100)

    args = parser.parse_args()
    set_seed(args.seed)
    batch_size = 8
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {args.device} backend")
    num_teachers = args.teacher_num
    train_transform = transforms.Compose([
        transforms.Resize(args.size),             # resize shortest side to 224 pixels
        transforms.CenterCrop(args.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
        ])
    test_transform = transforms.Compose([
        transforms.Resize(args.size),             # resize shortest side to 224 pixels
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
        ])
    # train_dataset = datasets.CIFAR100('./data/', train=True, transform=train_transform, download=True, )
    train_dataset = datasets.CIFAR100('./data/', train=True, transform=train_transform, download=True, )
    private_dataset = datasets.CIFAR100('./data/', train=False, transform=test_transform, download=True, )
    private_data_size = len(private_dataset)
    [private_label_data, test_data, private_unlabel_data] = random_split(private_dataset, [2000, 1000, private_data_size-3000],args.seed)
    # train
    total_size = len(train_dataset)
    lengths = [int(total_size/num_teachers)]*num_teachers
    lengths[-1] = total_size - int(total_size/num_teachers)*(num_teachers-1)
    teacher_datasets = torch.utils.data.random_split(train_dataset, lengths)
    all_private_dataloader = torch.utils.data.DataLoader(private_label_data, batch_size)
    teacher_preds_all = []
    train_accs = []
    test_accs = []
    for teacher in range(num_teachers):
        print("############################### Teacher {} Model Training #############################".format(teacher+1))
        train_acc, test_tacc, tr_model = train_model(teacher_datasets[teacher], private_label_data, args)
        train_accs.append(train_acc)
        test_accs.append(test_tacc)
        teacher_preds_all.append(predict_model(tr_model, all_private_dataloader, args.device))
        print("###########train:", min(train_accs), "," , max(train_accs), "test:", min(test_accs), "," , max(test_accs), "###########")
    ##################
    teacher_preds_all = torch.tensor(teacher_preds_all)
    print(teacher_preds_all.shape) 
    torch.save(teacher_preds_all, f"../teacher_preds/teacher_preds_cifar100.pth")  




if __name__ == "__main__":
    main()