import torch
import numpy as np
import argparse
from itertools import accumulate
from torchvision import datasets
from smooth_sensitivity_table import count_epsilon 

import medmnist
import PIL
from medmnist.info import INFO, HOMEPAGE, DEFAULT_ROOT
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from torchvision.datasets import ImageFolder

class mydataset(ImageFolder):
    def __init__(self, root, index = None, noisy_label = None, transform = None, target_transform = None, loader = ..., is_valid_file = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)
        if index is not None:
            self.samples = self.samples[index]
            self.targets = self.targets[index]
        if noisy_label is not None:
            self.targets = noisy_label
            
    def myloader(self, path: str):
        with open(path, "rb") as f:
            img = PIL.Image.open(f)
            return img.convert("RGB")
    
    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        target = self.targets[index]
        sample = self.myloader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(target)
        return sample, target
    
    def __len__(self) -> int:
        return len(self.targets)

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


class myEuroSAT(datasets.EuroSAT):
    def __init__(self, root: str, index = None, noisy_targets = None,transform = None, target_transform = None, download = False) -> None:
        super().__init__(root, transform, target_transform, download)
        self.index = index
        self.targets = np.array(self.targets)
        if self.index is not None:
            if noisy_targets is None:
                self.targets = self.targets[self.index]
            else:
                self.targets = noisy_targets


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        # path, _ = self.samples[index]
        if self.index is not None:
            path, _ = self.samples[self.index[idx]]
            target = self.targets[idx]
        else:
            path, target = self.samples[idx]
        sample = (self.loader(path))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
            

    def __len__(self) -> int:
        return len(self.targets)

class my_bloodmnist(medmnist.BloodMNIST):
    def __init__(self, split, index=None, noisy_targets=None, transform=None, target_transform=None, download=False, as_rgb=False, root=DEFAULT_ROOT):
        super().__init__(split, transform=transform, target_transform=target_transform, download=download, as_rgb=as_rgb, root=root)
        self.targets = np.array(self.labels).squeeze(1)
        self.index = index
        if index is not None:
            if noisy_targets is None:
                self.targets = self.targets[index]
            else:
                self.targets = np.array(noisy_targets)
    
    def __getitem__(self, idx):
        super().__getitem__(idx)
        if self.index is not None:
            img = self.imgs[self.index[idx]]
        else:
            img = self.imgs[idx]
        target = self.targets[idx].astype(int)
        img = PIL.Image.fromarray(img)
        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class my_dermamnist(medmnist.DermaMNIST):
    def __init__(self, split, index=None, noisy_targets=None, transform=None, target_transform=None, download=False, as_rgb=False, root=DEFAULT_ROOT):
        super().__init__(split, transform=transform, target_transform=target_transform, download=download, as_rgb=as_rgb, root=root)
        self.targets = np.array(self.labels).squeeze(1)
        self.index = index
        if index is not None:
            if noisy_targets is None:
                self.targets = self.targets[index]
            else:
                self.targets = np.array(noisy_targets)
    
    def __getitem__(self, idx):
        super().__getitem__(idx)
        if self.index is not None:
            img = self.imgs[self.index[idx]]
        else:
            img = self.imgs[idx]
        target = self.targets[idx].astype(int)
        img = PIL.Image.fromarray(img)
        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class my_pathmnist(medmnist.PathMNIST):
    def __init__(self, split, index=None, noisy_targets=None, transform=None, target_transform=None, download=False, as_rgb=False, root=DEFAULT_ROOT):
        super().__init__(split, transform=transform, target_transform=target_transform, download=download, as_rgb=as_rgb, root=root)
        self.targets = np.array(self.labels).squeeze(1)
        self.index = index
        if index is not None:
            if noisy_targets is None:
                self.targets = self.targets[index]
            else:
                self.targets = np.array(noisy_targets)
    
    def __getitem__(self, idx):
        super().__getitem__(idx)
        if self.index is not None:
            img = self.imgs[self.index[idx]]
        else:
            img = self.imgs[idx]
        target = self.targets[idx].astype(int)
        img = PIL.Image.fromarray(img)
        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class my_tissuemnist(medmnist.TissueMNIST):
    def __init__(self, split, index=None, noisy_targets=None, transform=None, target_transform=None, download=False, as_rgb=False, root=DEFAULT_ROOT):
        super().__init__(split, transform=transform, target_transform=target_transform, download=download, as_rgb=as_rgb, root=root)
        self.targets = np.array(self.labels).squeeze(1)
        self.index = index
        if index is not None:
            if noisy_targets is None:
                self.targets = self.targets[index]
            else:
                self.targets = np.array(noisy_targets)
    
    def __getitem__(self, idx):
        super().__getitem__(idx)
        if self.index is not None:
            img = self.imgs[self.index[idx]]
        else:
            img = self.imgs[idx]
        target = self.targets[idx].astype(int)
        img = PIL.Image.fromarray(img)
        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class Celeba_gender(datasets.CelebA):
    def __init__(self, root, index = None ,noisy_targets=None,split="train", target_type="attr", transform = None, target_transform = None, download = True) -> None:
        super().__init__(root, split=split, target_type=target_type, transform=transform, target_transform=target_transform, download=download)
        self.targets = np.array(self.attr[:,20])
        self.index = index
        # self.data = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        if index is not None:
            # self.data = self.data[index];
            if noisy_targets is None:
                self.targets = self.targets[index]
            else:
                self.targets = np.array(noisy_targets)
    def __getitem__(self, i):
        target = self.targets[i]
        if self.index is not None:
            img = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[self.index[i]]))
        else :
            img = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[i]))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        super().__len__()
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.attr)


def noisy_aggregation(true_label, teacher_preds, args):
    np.random.seed(args.seed)
    total_noisy_labels = torch.tensor([-1 for i in range(len(true_label))])
    noisy_labels = list()
    sigma1 = args.sigma1
    sigma2 = args.sigma2
    big_t = args.threshold
    num_class = args.class_num
    private_data_size = args.queries
    for image_i in range(private_data_size):
        label_counts = torch.bincount(torch.from_numpy(teacher_preds[:, image_i]), minlength=num_class).float()
        max_x = max(label_counts)
        # print(max_x)
        if max_x + np.random.normal(0, sigma1) > big_t:
            for i in range(len(label_counts)):
            # label_counts[i] += np.random.laplace(0, beta, 1)[0]
                label_counts[i] += np.random.normal(0, sigma2)
            noisy_label = torch.argmax(label_counts)
        else:
            noisy_label = -1
        # noisy_label = torch.argmax(label_counts)
        noisy_labels.append(noisy_label)
    noisy_labels = torch.tensor(noisy_labels)
    ans_num = 0
    noisy_acc = 0
    for i in range(private_data_size):
        if(noisy_labels[i] != -1):
            ans_num += 1
            if noisy_labels[i] == true_label[i]:
                noisy_acc += 1
    print("Querires num:",private_data_size)
    print("Ans num:", ans_num)
    print("Noisy label accuracy:", noisy_acc/ans_num)
    # print(noisy_labels)
    total_noisy_labels[:private_data_size] = noisy_labels
    # print(total_noisy_labels)
    torch.save(total_noisy_labels,f"./noisy_label/nosiy_labels_{args.dataset}_{args.preds_file}_{args.eps}.pth")
    # return noisy_labels


def random_split(lengths, seed):
    torch.manual_seed(seed)
    indices = torch.randperm(sum(lengths)).tolist()
    return [indices[offset - length:offset] for offset, length in zip(accumulate(lengths), lengths)]

def count_num(input_mat, class_num):
    num_teachers, n = input_mat.shape
    counts_mat = np.zeros((n, class_num)).astype(np.int32)
    for i in range(n):
        for j in range(num_teachers):
            counts_mat[i, int(input_mat[j, i])] += 1
        
    np.save(f"./preds_count.npy", counts_mat)

def main():
    parser = argparse.ArgumentParser(description='dif epsilon')
    parser.add_argument('--seed', type=int, default=8872574) #4
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--querires_num', type=int, default=1000)
    parser.add_argument('--sigma1', type=float, default=150)
    parser.add_argument('--sigma2', type=float, default=40)
    parser.add_argument('--threshold', type=float, default=200)
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--queries', type=int, default=100)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--preds_file', type=str, default='swin_1000_192')
    parser.add_argument('--baseline_file', type=str, default=None)
    parser.add_argument('--data_independent', type=bool, default=False)
    parser.add_argument('--order', type=float, default=None)
    parser.add_argument('--teachers', type=int, default=None)
    parser.add_argument('--counts_file', type=str, default="./preds_count.npy")
    parser.add_argument('--eps', type=str, default="1")

    args = parser.parse_args()

    if args.dataset == 'CIFAR10':
        dataset = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
        datasize = len(dataset)
        [label_index, test_index, unlabel_index] = random_split([2000, 1000, datasize-3000], args.seed)
    if args.dataset == 'EuroSAT':
        dataset = myEuroSAT('./data/EuroSAT',download=True)
        datasize = len(dataset)
        [train_index, semi_test_index, label_index,semi_ulabel_index] = random_split([10000, 1000, 1000, datasize-12000],args.seed)
    if args.dataset == 'BloodMNIST':
        dataset = my_bloodmnist(split='test', download=True)
        datasize = len(dataset)
        [label_index, test_index, unlabel_index] = random_split([1000, 1000, datasize-2000], args.seed)
    if args.dataset == 'CIFAR100':
        dataset = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)
        datasize = len(dataset)
        [label_index, test_index, unlabel_index] = random_split([2000, 1000, datasize-3000], args.seed)
    if args.dataset == 'TissueMNIST':
        dataset = my_tissuemnist(split='test', download=True)
        datasize = len(dataset)
        [label_index, test_index, unlabel_index] = random_split([2000, 1000, datasize-3000], args.seed)
    if args.dataset == 'CelebA':
        dataset = Celeba_gender('./data/CelebA', split='test',  download=True)
        datasize = len(dataset)
        [label_index, test_index, unlabel_index] = random_split([3000, 1000, datasize-4000], args.seed)
    if args.dataset == 'SVHN':
        dataset = mySVHN('./data/SVHN', split='test', download=True)    
        datasize = len(dataset)
        [label_index, test_index, unlabel_index] = random_split([2000, 1000, datasize-3000], args.seed)
    if args.dataset == 'PathMNIST':
        dataset = my_pathmnist(split='test', download=True)
        datasize = len(dataset)
        [label_index, test_index, unlabel_index] = random_split([1000, 1000, datasize-2000], args.seed)
    if args.dataset == 'DermaMNIST':
        dataset = my_dermamnist(split='test', download=True)
        datasize = len(dataset)
        [label_index, test_index, unlabel_index] = random_split([1000, 1000, datasize-2000], args.seed)
    if args.dataset == 'FFHQ':
        dataset = mydataset('/root/autodl-tmp/gender/')
        datasize = len(dataset)
        [train_set,unlabel_data, label_index, test_set] = random_split([50000, datasize-53000, 2000, 1000],args.seed)



    ## fan dataset
    # dataset = datasets.CIFAR10('./data/CIFAR10', train=False, download=True, )
    # dataset = myEuroSAT('./data/EuroSAT',download=True)
    # dataset = my_medmnist(split='test', download=True)
    # [label_index, test_index, unlabel_index] = random_split([2000, 1000, datasize-3000], args.seed)
    # [train_index, semi_test_index, label_index,semi_ulabel_index] = random_split([10000, 1000, 1000, datasize-12000],args.seed)
    # [label_index, test_index, unlabel_index] = random_split([2000, 1000, datasize-3000], args.seed)
    # preds = torch.load(f"./newdata/teacher_preds_1000_192_192.pth").numpy()



    preds = torch.load(f"./teacher_preds/teacher_preds_{args.preds_file}.pth").numpy()
    count_num(preds, args.class_num)
    count_epsilon(args)
    target = np.array(dataset.targets)
    true_label = target[label_index]
    noisy_aggregation(true_label, preds, args)

    


if __name__ == "__main__":
    main()