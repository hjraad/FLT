#----------------------------------------------------------------------------
# Created By  : Mohammad Abdizadeh & Hadi Jamali-Rad
# Created Date: 23-Nov-2020
# 
# Refactored By: Sayak Mukherjee
# Last Update: 27-Oct-2023
# ---------------------------------------------------------------------------
# File contains for implementation of FEMNIST dataset.
# ---------------------------------------------------------------------------

# import os
# import json
# import warnings
# import numpy as np

# from PIL import Image
# from collections import defaultdict
# from pathlib import Path
# from torchvision.datasets.vision import VisionDataset
# from torchvision.datasets.utils import download_and_extract_archive

import errno
import os.path
import torch
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets import MNIST, utils

# def read_dir(data_dir):
#     clients = []
#     groups = []
#     data = defaultdict(lambda : None)

#     files = os.listdir(data_dir)
#     files = [f for f in files if f.endswith('.json')]
#     for f in files:
#         print(f)
#         file_path = os.path.join(data_dir,f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         clients.extend(cdata['users'])
#         if 'hierarchies' in cdata:
#             groups.extend(cdata['hierarchies'])
#         data.update(cdata['user_data'])

#     clients = list(sorted(data.keys()))
#     return clients, groups, data

# def read_data(train_data_dir, test_data_dir, train_flag):
#     '''parses data in given train and test data directories

#     assumes:
#     - the data in the input directories are .json files with 
#         keys 'users' and 'user_data'
#     - the set of train set users is the same as the set of test set users
    
#     Return:
#         clients: list of client ids
#         groups: list of group ids; empty list if none found
#         train_data: dictionary of train data
#         test_data: dictionary of test data
#     '''
#     if train_flag == True:
#         clients, groups, data = read_dir(train_data_dir)
#     else:
#         clients, groups, data = read_dir(test_data_dir)

#     #assert train_clients == test_clients
#     #assert train_groups == test_groups

#     return clients, groups, data

# class FEMNIST3(VisionDataset):
#     """
#     This dataset is derived from the Leaf repository
#     (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
#     dataset, grouping examples by writer. Details about Leaf were published in
#     "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

#     Args:
#         root (string): Root directory of dataset where ``MNIST/processed/training.pt``
#             and  ``FEMNIST/processed/test.pt`` exist.
#         train (bool, optional): If True, creates dataset from ``training.pt``,
#             otherwise from ``test.pt``.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#     """

#     resources = [
#         ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
#          '59c65cec646fc57fe92d27d83afdf0ed')
#     ]

#     training_file = 'training.pt'
#     test_file = 'test.pt'
#     classes =  ['0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',
#                     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
#                     'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',  'Z',
#                     'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
    
    
#     @property
#     def train_labels(self):
#         warnings.warn("train_labels has been renamed targets")
#         return self.targets

#     @property
#     def test_labels(self):
#         warnings.warn("test_labels has been renamed targets")
#         return self.targets

#     @property
#     def train_data(self):
#         warnings.warn("train_data has been renamed data")
#         return self.data

#     @property
#     def test_data(self):
#         warnings.warn("test_data has been renamed data")
#         return self.data

#     def __init__(self, root, train=True, transform=None, target_transform=None,
#                  download=False):
#         super(FEMNIST, self).__init__(root, transform=transform,
#                                     target_transform=target_transform)
#         self.train = train  # training set or test set

#         if download:
#             self.download()

#         if not self._check_exists():
#             raise RuntimeError('Dataset not found.' +
#                                ' You can use download=True to download it')

#         train_data_dir = os.path.join(root, 'femnist', 'FEMNIST', '', 'train')
#         test_data_dir = os.path.join(root, 'femnist', 'FEMNIST', 'test')
#         self.dict_users = {}

#         if self.train == True:
#             self.users, groups, self.data = read_data(train_data_dir, test_data_dir, train_flag = True)
#         else:
#             self.users, groups, self.data = read_data(train_data_dir, test_data_dir, train_flag = False)

#         class_names_map = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
#             10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35, 
#             36,  37,  12,  38,  39,  40,  41,  42,  18,  19,  20,  21,  22,  43,  24,  25,  44,  45,  28,  46,  30,  31,  32,  33,  34,  35]
#         # TODO: automate this
#         if False:# 47 classess
#             for i in range(len(self.users)):
#                 for j in range(len(self.data[self.users[i]]['y'])):
#                     ll = self.data[self.users[i]]['y'][j]
#                     self.data[self.users[i]]['y'][j] = class_names_map[ll]

#         counter = 0        
#         for i in range(len(self.users)):
#             lst = list(counter + np.arange(len(self.data[self.users[i]]['y'])))
#             self.dict_users.update({i: set(lst)})
#             counter = lst[-1] + 1


#         self.dict_index = {}# define a dictionary to keep the location of a sample and the corresponding
#         length_data = 0
#         for i in range(len(self.users)):
#             for j in range(len(self.data[self.users[i]]['y'])):
#                 self.dict_index[length_data] = [i, j]
#                 length_data += 1
#         self.length_data = length_data

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         [i, j] = self.dict_index[index]
#         img, target = self.data[self.users[i]]['x'][j], int(self.data[self.users[i]]['y'][j])

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(np.array(img).reshape(28,28))

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return self.length_data

#     @property
#     def raw_folder(self):
#         return os.path.join(self.root, self.__class__.__name__, 'raw')

#     @property
#     def processed_folder(self):
#         return os.path.join(self.root, self.__class__.__name__, 'processed')

#     @property
#     def class_to_idx(self):
#         return {_class: i for i, _class in enumerate(self.classes)}

#     def _check_exists(self):
#         return (os.path.exists(os.path.join(self.processed_folder,
#                                             self.training_file)) and
#                 os.path.exists(os.path.join(self.processed_folder,
#                                             self.test_file)))

#     def download(self):
#         """Download the MNIST data if it doesn't exist in processed_folder already."""

#         if self._check_exists():
#             return

#         os.makedirs(self.raw_folder, exist_ok=True)
#         os.makedirs(self.processed_folder, exist_ok=True)

#         # download files
#         for url, md5 in self.resources:
#             filename = url.rpartition('/')[2]
#             download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

#         # process and save as torch files
#         print('Processing...')
#         """
#         training_set = (
#             read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
#             read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
#         )
#         test_set = (
#             read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
#             read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
#         )
#         with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
#             torch.save(training_set, f)
#         with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
#             torch.save(test_set, f)
#         """
#         os.replace(os.path.join(self.raw_folder, 'training.pt'), os.path.join(self.processed_folder, 'training.pt'))
#         os.replace(os.path.join(self.raw_folder, 'test.pt'), os.path.join(self.processed_folder, 'test.pt'))

#         print('Done!')

#     def extra_repr(self):
#         return "Split: {}".format("Train" if self.train is True else "Test")

class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_legacy_exist():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

        self.dict_users = dict()
        
        counter = 0
        for i in range(len(self.users_index)):
            lst = list(counter + np.arange(self.users_index[i]))
            self.dict_users[i] = lst
            counter = lst[-1] + 1

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def makedir_exist_ok(self, dirpath):
        """
        Python2 support for os.makedirs(.., exist_ok=True)
        """
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            utils.check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""

        if self._check_legacy_exist():
            return

        self.makedir_exist_ok(self.raw_folder)
        self.makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)