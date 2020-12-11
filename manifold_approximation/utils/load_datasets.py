'''
load the datasets
'''
import torch, torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets
from emnist import list_datasets, extract_training_samples, extract_test_samples

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
        
        
    else:
        pass

    image_datasets = {'train': train_data, 'test': test_data}
    dataloaders = {x: DataLoader(image_datasets[x], 
                                batch_size=batch_size,
                                shuffle=shuffle_flag, 
                                num_workers=4)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, image_datasets, dataset_sizes, class_names