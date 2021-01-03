import os
from glob import glob

def extract_model_name(model_root_dir, dataset_name):
    aa = glob(f'{model_root_dir}/*-{dataset_name}-*_best.pt')
    aa.sort(reverse=True)
    if aa:
        return aa[0].split('/')[-1].split('_best')[0]
    else:
        exit('no relevant model could be found!')