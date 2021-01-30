'''
load the datasets
'''
import numpy as np
import torch, torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import datasets
from torchvision.datasets.vision import VisionDataset
from utils.model_utils import read_data
import os
from PIL import Image
# from emnist import list_datasets, extract_training_samples, extract_test_samples

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


class FEMNIST(VisionDataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``FEMNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes =  ['0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                    'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',  'Z',
                    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
    
    
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(FEMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        train_data_dir = os.path.join('..', 'data', 'femnist', 'FEMNIST', 'train')
        test_data_dir = os.path.join('..', 'data', 'femnist', 'FEMNIST', 'test')  
        self.dict_users = {}

        if self.train == True:
            self.users, groups, self.data = read_data(train_data_dir, test_data_dir, train_flag = True)
        else:
            self.users, groups, self.data = read_data(train_data_dir, test_data_dir, train_flag = False)

        class_names_map = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
            10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35, 
            36,  37,  12,  38,  39,  40,  41,  42,  18,  19,  20,  21,  22,  43,  24,  25,  44,  45,  28,  46,  30,  31,  32,  33,  34,  35]
        # TODO: automate this
        if True:# 47 classess
            for i in range(len(self.users)):
                for j in range(len(self.data0[self.users[i]]['y'])):
                    ll = self.data0[self.users[i]]['y'][j]
                    self.data0[self.users[i]]['y'][j] = class_names_map[ll]

        counter = 0        
        for i in range(len(self.users)):
            lst = list(counter + np.arange(len(self.data[self.users[i]]['y'])))
            self.dict_users.update({i: set(lst)})
            counter = lst[-1] + 1


        self.dict_index = {}# define a dictionary to keep the location of a sample and the corresponding
        length_data = 0
        for i in range(len(self.users)):
            for j in range(len(self.data[self.users[i]]['y'])):
                self.dict_index[length_data] = [i, j]
                length_data += 1
        self.length_data = length_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        [i, j] = self.dict_index[index]
        img, target = self.data[self.users[i]]['x'][j], int(self.data[self.users[i]]['y'][j])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.array(img).reshape(28,28))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length_data

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        """
        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        """
        os.replace(os.path.join(self.raw_folder, 'training.pt'), os.path.join(self.processed_folder, 'training.pt'))
        os.replace(os.path.join(self.raw_folder, 'test.pt'), os.path.join(self.processed_folder, 'test.pt'))

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


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
        
    elif dataset_name == 'CIFAR110':
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
    
    elif dataset_name == 'FEMNIST':        
        train_data = FEMNIST(data_root_dir + '/femnist/', train=True, download=True, transform=transforms_dict['train'])
        test_data = FEMNIST(data_root_dir + '/femnist/', train=False, download=True, transform=transforms_dict['test'])     

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
    elif dataset_name == 'CIFAR110': #TODO: fix this in a smarter way; order matters here! 
        class_names = train_data_10.classes + train_data_o.classes[10:] 
    else:
        class_names = image_datasets['train'].classes
    
    return dataloaders, image_datasets, dataset_sizes, class_names