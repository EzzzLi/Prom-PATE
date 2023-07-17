import torch
import torchvision
import argparse
import PIL
import os
import numpy as np
from torch import nn, optim
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from itertools import accumulate
from torch.utils.data import Subset
import copy

mean_svhn = (0.4380, 0.4440, 0.4730)
std_svhn = (0.1751, 0.1771, 0.1744)




class mySVHN(datasets.SVHN):
    def __init__(self, root, index = None, noisy_targets = None,split = "train", transform= None, target_transform = None, download = False) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.labels = np.array(self.labels)
        self.index = index
        self.targets = self.labels
        if index is not None:
            if noisy_targets is None:
                self.targets = self.labels[index]
            else:
                self.targets = np.array(noisy_targets)

    def __getitem__(self, i):
        target = self.targets[i]
        if self.index is not None:
            img = self.data[self.index[i]]
        else:
            img = self.data[i]
        img = PIL.Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        # return super().__getitem__(index)

    def __len__(self) -> int:
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.data)
        # return super().__len__()

class transfer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.wide_resnet50_2(pretrained=True)
        self.model.eval()
        for pram in self.model.parameters():
            pram.requires_grad = False
        self.new_layers = nn.Sequential(nn.ReLU(),nn.Linear(1000, 10))
    def forward(self, t):
        t = self.model(t)
        t = self.new_layers(t)
        return t        

## this is reprogramming code
class reProgrammingNetwork(nn.Module):
    def __init__(self, input_size=224, patch_H_size=192, patch_W_size=192, channel_out=3, device="cpu") -> None:
        super().__init__()
        self.device = device
        self.channel_out = channel_out
        self.input_size = input_size
        self.pre_model = torchvision.models.wide_resnet50_2(pretrained=True)
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
        self.new_layers = nn.Sequential(nn.ReLU(),nn.Linear(1000, 10))

    def hg(self, imagenet_label):
        return imagenet_label[:,:10]
    
    def forward(self, image):
        X = torch.zeros(image.shape[0], self.channel_out, self.input_size, self.input_size)
        X[:,:,self.H_start:self.H_end,self.W_start:self.W_end] = image.repeat(1,1,1,1).data.clone()
        X = Parameter(X, requires_grad=True).to(self.device)

        P = torch.tanh(self.W * self.M)
        X_adv = P + X
        Y_adv = self.pre_model(X_adv)
        Y = self.new_layers(Y_adv)
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
    model = reProgrammingNetwork(device=device).to(device)
    # model = transfer().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
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
            label_hat = model(image)
            loss = loss_function(label_hat, label)
            train_loss += loss.item()
            train_acc += sum(label.cpu().numpy() == label_hat.data.cpu().numpy().argmax(1))
            loss.backward()
            optimizer.step()
            
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


def main():
    parser = argparse.ArgumentParser(description='pate train')
    parser.add_argument('--seed', type=int, default=8872574) #4
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--teacher_num', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--LR_step', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.7)

    args = parser.parse_args()
    set_seed(args.seed)
    batch_size = 16
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {args.device} backend")
    num_teachers = args.teacher_num
    train_transform = transforms.Compose([
        transforms.Resize(192),
        transforms.RandomCrop(192, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_svhn, std_svhn),
        ])
    test_transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
        transforms.Normalize(mean_svhn, std_svhn),
        ])
    train_dataset = mySVHN('../data/SVHN', split='train', transform=train_transform, download=True, )
    private_dataset = mySVHN('../data/SVHN', split='test', transform=test_transform, download=True, )
    private_data_size = len(private_dataset)
    print(private_data_size)

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
    torch.save(teacher_preds_all, f"../teacher_preds/teacher_preds_svhn.pth")  




if __name__ == "__main__":
    main()