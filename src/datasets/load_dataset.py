#----------------------------------------------------------------------------
# Created By  : Mohammad Abdizadeh & Hadi Jamali-Rad
# Created Date: 23-Nov-2020
# 
# Refactored By: Sayak Mukherjee
# Last Update: 27-Oct-2023
# ---------------------------------------------------------------------------
# File contains for loading datasets for the experiments.
# ---------------------------------------------------------------------------

import os
import numpy as np
import torch, torchvision

from urllib.error import URLError
from .femnist import FEMNIST
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

EMNIST_RESOURCES = [
        ("https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip", 
         "58c8d27c78d21e728a6bc7b3cc06412e"),
    ]

class MySubset(Dataset):
    '''
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    '''
    def __init__(self, dataset, indices, labels):
        self.dataset = dataset
        self.indices = indices
        #labels_hold = torch.ones(len(dataset)).type(torch.long) *300 #( some number not present in the #labels just to make sure
        labels_hold = torch.ones(len(dataset), dtype=int)
        # labels_hold[self.indices] = labels 
        labels_hold[self.indices] = torch.from_numpy(np.array(labels, dtype=int))
        self.labels = labels_hold
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.labels[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)
    
def download_emnist(root) -> None:
    """Download the EMNIST data if it doesn't exist already."""

    def _check_exists() -> bool:
        return all(
            check_integrity(os.path.join(raw_folder(), os.path.splitext(os.path.basename(url))[0]))
            for url, _ in EMNIST_RESOURCES
        )
    
    def raw_folder() -> str:
        return os.path.join(root, "EMNIST", "raw")

    if _check_exists():
        return

    os.makedirs(raw_folder(), exist_ok=True)

    for url, md5 in EMNIST_RESOURCES:
        filename = os.path.basename(url)
        try:
            download_and_extract_archive(url, download_root=raw_folder(), filename=filename, md5=md5)
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")
            continue
        finally:
            print()

    # download_and_extract_archive(EMNIST_RESOURCES[0][0], download_root=raw_folder(), md5=EMNIST_RESOURCES[0][1])
    # gzip_folder = os.path.join(self.raw_folder, "gzip")
    # for gzip_file in os.listdir(gzip_folder):
    #     if gzip_file.endswith(".gz"):
    #         extract_archive(os.path.join(gzip_folder, gzip_file), self.raw_folder)
    # shutil.rmtree(gzip_folder)

def load_dataset(dataset_name, data_root_dir, dataset_split=''):
    '''
    Paramters: 
        dataset_name
        data_root_dir
        dataset_split: if a dataset has different splits/types, e.g. EMNIST has byclass, bymerge, digits, balanced, etc.
        
    Returns:
        image_datasets: dict of image datasets
        dataset_sizes
        class_names
    '''

    if dataset_name in ['CIFAR10', 'CIFAR20', 'CIFAR100']:
        transforms_dict = {    
        'train': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'test': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        }

    else:  
        transforms_dict = {
            'train': transforms.Compose([transforms.ToTensor()]),
            'test': transforms.Compose([transforms.ToTensor()])
        }
    


    if dataset_name == 'EMNIST': # torchvision version seems to have a bug with class mapping (so trasnforms is hacked)

        # Fix to URL issue in EMNIST
        download_emnist(data_root_dir)

        train_data = datasets.EMNIST(root=data_root_dir, split=dataset_split, 
                                    train=True, 
                                    transform=torchvision.transforms.Compose([
                                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                                    lambda img: torchvision.transforms.functional.hflip(img),
                                    torchvision.transforms.ToTensor()]))

        test_data = datasets.EMNIST(root=data_root_dir, split=dataset_split, 
                                    train=False, 
                                    transform=torchvision.transforms.Compose([
                                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                                    lambda img: torchvision.transforms.functional.hflip(img),
                                    torchvision.transforms.ToTensor()]))
        
        # EMNIST using installed emnist package
        # train_data = extract_training_samples('balanced')
        # test_data = extract_training_samples('balanced')
        # class_names = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 'A', 'B', 'C', 'D', 'E', 'F', 
        #                'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 
        #                'U', 'V', 'W', 'X', 'Y',  'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

    elif dataset_name == 'MNIST':
        train_data = datasets.MNIST(root=data_root_dir, train=True, 
                                     download=True, transform=transforms_dict['train'])

        test_data = datasets.MNIST(root=data_root_dir, train=False, 
                                    download=True, transform=transforms_dict['test'])
        
    elif dataset_name == 'FMNIST':
        train_data = datasets.FashionMNIST(root=data_root_dir, train=True, 
                                     download=True, transform=transforms_dict['train'])

        test_data = datasets.FashionMNIST(root=data_root_dir, train=False, 
                                    download=True, transform=transforms_dict['test'])
        
        
    elif dataset_name == 'CIFAR10':
        train_data = datasets.CIFAR10(root=data_root_dir, train=True, 
                                     download=True, transform=transforms_dict['train'])

        test_data = datasets.CIFAR10(root=data_root_dir, train=False, 
                                    download=True, transform=transforms_dict['test'])
        
    elif dataset_name == 'CIFAR100':
        train_data_o = datasets.CIFAR100(root=data_root_dir, train=True, 
                                     download=True, transform=transforms_dict['train'])

        test_data_o = datasets.CIFAR100(root=data_root_dir, train=False, 
                                    download=True, transform=transforms_dict['test'])
        
        # subselect 20 classes out of 100 and create a dataset accordingly
        selected_dict = {3:'bear',
                         8:'bicycle', 
                         13:'bus', 
                         15:'camel',
                         20:'chair', 
                         22:'clock', 
                         25:'couch',
                         29:'dinosaur',
                         31:'elephant',
                         37:'house',
                         43:'lion',
                         46:'man',
                         51:'mushroom',
                         54:'orchid',
                         65:'rabbit',
                         84:'table',
                         85:'tank',
                         86:'telephone',
                         90:'train',
                         92:'tulip'
                        }
    
        # Get all targets
        targets_train = train_data_o.targets
        targets_test = test_data_o.targets

        # Specify which class to keep from train
        classidx_to_keep = list(selected_dict.keys())
        
        # Get indices to keep from train split
        idx_to_keep_train = [ind for (ind, target) in enumerate(targets_train) if target in classidx_to_keep] 
        idx_to_keep_test = [ind for (ind, target) in enumerate(targets_test) if target in classidx_to_keep] 
        
        # Only keep your desired classes
        targets_train = np.array(targets_train)[np.array(idx_to_keep_train)]
        targets_test = np.array(targets_test)[np.array(idx_to_keep_test)]
        
        # train_data = MySubset(train_data, list(idx_to_keep_train), list(targets_train))
        
        train_data = Subset(train_data_o, idx_to_keep_train)
        test_data = Subset(test_data_o, idx_to_keep_test)
        
    elif dataset_name == 'CIFAR20':
        train_data_o = datasets.CIFAR100(root=data_root_dir, train=True, 
                                     download=True, transform=transforms_dict['train'])

        test_data_o = datasets.CIFAR100(root=data_root_dir, train=False, 
                                    download=True, transform=transforms_dict['test'])
        
        # subselect 10 classes out of 100 (all after 9) and create a dataset accordingly
        selected_dict = {11:'boy', 
                         20:'chair', 
                         22:'clock', 
                         31:'elephant',
                         37:'house',
                         51:'mushroom',
                         54:'orchid',
                         65:'rabbit',
                         86:'telephone',
                         90:'train',
                        }
    
        # Get all targets
        targets_train = train_data_o.targets
        targets_test = test_data_o.targets

        # Specify which class to keep from train
        classidx_to_keep = list(selected_dict.keys())
        
        # Get indices to keep from train split
        idx_to_keep_train = [ind for (ind, target) in enumerate(targets_train) if target in classidx_to_keep] 
        idx_to_keep_test = [ind for (ind, target) in enumerate(targets_test) if target in classidx_to_keep] 
        
        # Only keep your desired classes
        targets_train = np.array(targets_train)[np.array(idx_to_keep_train)]
        targets_test = np.array(targets_test)[np.array(idx_to_keep_test)]
        
        train_data_100 = Subset(train_data_o, idx_to_keep_train)
        test_data_100 = Subset(test_data_o, idx_to_keep_test)
        
        # the rest up to 20 comes from CIFAR10
        train_data_10 = datasets.CIFAR10(root=data_root_dir, train=True, 
                                     download=True, transform=transforms_dict['train'])

        test_data_10 = datasets.CIFAR10(root=data_root_dir, train=False, 
                                    download=True, transform=transforms_dict['test'])
        
        train_data = ConcatDataset([train_data_100, train_data_10])
        test_data = ConcatDataset([test_data_100, test_data_10])
        
    elif dataset_name == 'CINIC10':
        train_data = datasets.ImageFolder(root=data_root_dir + '/cinic-10/train', transform=transforms_dict['train'])
        test_data = datasets.ImageFolder(root=data_root_dir + '/cinic-10/test', transform=transforms_dict['test'])
    
    # Dataset to use LEAF data
    # Data should be put under data/femnist/FEMNIST/train and data/femnist/FEMNIST/test
    # For comparing with baselines only use _0 and _1 files both in train and test folders
    elif dataset_name == 'FEMNIST':        
        train_data = FEMNIST(data_root_dir + '/femnist/', train=True, download=True, transform=transforms_dict['train'])
        test_data = FEMNIST(data_root_dir + '/femnist/', train=False, download=True, transform=transforms_dict['test'])     

    else:
        raise ValueError(f'{dataset_name} not implemented.')

    image_datasets = {'train': train_data, 'test': test_data}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    if dataset_name == 'CIFAR100':
        class_names = train_data_o.classes
    elif dataset_name == 'CIFAR20': #TODO: fix this in a smarter way; order matters here! 
        class_names = train_data_10.classes + classidx_to_keep 
    else:
        class_names = image_datasets['train'].classes


    return image_datasets, dataset_sizes, class_names