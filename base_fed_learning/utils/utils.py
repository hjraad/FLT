import os
from glob import glob
from collections import UserList

def extract_model_name(model_root_dir, dataset_name):
    aa = glob(f'{model_root_dir}/*-{dataset_name}-*_best.pt')
    aa.sort(reverse=True)
    if aa:
        return aa[0].split('/')[-1].split('_best')[0]
    else:
        exit('no relevant model could be found!')

class ModelContainer(UserList):
    def __init__(self, *args):
        if args:
            super(ModelContainer, self).__init__(args[0])
        else:
            super(ModelContainer, self).__init__()
            
    def __getitem__(self, index):
        return self.data[index].to(self.device)

    def __setitem__(self, index, value):
        self.data[index] = value.to('cpu')

    def get_on_device(self, index, device):
        return self.data[index].to(device)

    def set_device(self, device):
        self.device = device


import torch
if __name__ == '__main__':
    mc = ModelContainer()
    mc.set_device('cuda:0')
    for i in range(1000):
        mc.append(torch.randint(1,10,(3,)))
    print('list populated')
    for i in range(1000):
        print(mc[i], mc[i].device)