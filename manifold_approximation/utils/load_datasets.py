'''
load the datasets
'''
import numpy as np
import torch, torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from emnist import list_datasets, extract_training_samples, extract_test_samples

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

def load_dataset(dataset_name, data_root_dir, transforms_dict, batch_size=8, shuffle_flag=False, dataset_split=''):
    '''
    Paramters: 
        dataset_name
        transform_dict: a dictionary of transformations for train, test
        batch_size
        dataset_split: if a dataset has different splits/types, e.g. EMNIST has byclass, bymerge, digits, balanced, etc.
        
    Returns:
        dataloaders: dict of dataloaders
        image_datasets: dict of image datasets
        dataset_sizes
        class_names
    '''
    if dataset_name == 'EMNIST': # torchvision version seems to have a bug with class mapping (so trasnforms is hacked)
        train_data = datasets.EMNIST(root=data_root_dir, split=dataset_split, 
                                                train=True, download=True, 
                                                transform=torchvision.transforms.Compose([
                                                lambda img: torchvision.transforms.functional.rotate(img, -90),
                                                lambda img: torchvision.transforms.functional.hflip(img),
                                                torchvision.transforms.ToTensor()]))

        test_data = datasets.EMNIST(root=data_root_dir, split=dataset_split, 
                                                    train=False, download=True, 
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
                         37:'couse',
                         43:'lion',
                         46:'man',
                         51:'mushroom',
                         54:'orchid',
                         65:'rabit',
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
        
        
    elif dataset_name == 'CINIC10':
        train_data = datasets.ImageFolder(root=data_root_dir + '/cinic-10/train', transform=transforms_dict['train'])

        test_data = datasets.ImageFolder(root=data_root_dir + '/cinic-10/test', transform=transforms_dict['test'])
         
    else:
        pass

    image_datasets = {'train': train_data, 'test': test_data}
    dataloaders = {x: DataLoader(image_datasets[x], 
                                batch_size=batch_size,
                                shuffle=shuffle_flag, 
                                num_workers=4)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    
    if dataset_name == 'CIFAR100':
        class_names = train_data_o.classes
    else:
        class_names = image_datasets['train'].classes
    
    return dataloaders, image_datasets, dataset_sizes, class_names