import matplotlib
# matplotlib.use('Agg')
import sys
sys.path.append("./../")
sys.path.append("./../../")
sys.path.append("./")

import os
import matplotlib.pyplot as plt
from utils.options import args_parser
import argparse
from torchvision import datasets, transforms
import torch
import torchvision

import pandas as pd

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
     # ----------------------------------
    plt.close('all')

    results_path = f'{args.results_root_dir}/main_fed/'
    entries = os.listdir(results_path)
    
    # plot loss curve
    fig, ax = plt.subplots()
    line_style = ['k--', 'k:','r--', 'r:','b--', 'b:']

    for (idx, entry) in enumerate(entries):
        if not entry.endswith(".csv"):
            continue

        method = ''
        print(idx, entry)
        index0 = entry.find('method')
        index1 = entry.find('multicenter')

        if entry[index1+12] != '0':
            method = 'multicenter'
        else:
            method = entry[index0+7:index1-1]
        
        print(f'clustering method is {method}')

        df = pd.read_csv(f'{results_path}/{entry}')

        ax.plot(range(len(df['training_accuracy'])), df['training_accuracy'], line_style[2*idx], label=f'{method}: train_accuracy')
        ax.plot(range(len(df['test_accuracy'])), df['test_accuracy'], line_style[2*idx+1], label=f'{method}: test_accuracy')

    legend = ax.legend(loc='lower right')#, shadow=True, fontsize='x-large')    
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
