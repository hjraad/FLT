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
import numpy as np
import pandas as pd
from glob import glob


def visualize(result_directory_name):
    # plotting settings
    plot_linewidth = 1.5
    text_size = 5
    marker_step = 5
    marker_size = 5
    legend_text_size = 'x-large'
    legend_linewidth = 1.5
    grid_ticks = np.arange(0, 101, 10)
    
    name_dict = {
        'fedavg': 'FedAvg',
        'local': 'Local',
        'fedsem': 'FedSem',
        'ucfl-enc1': 'Ours (Enc1)',
        'ucfl-enc2': 'Ours (Enc2)'
    }

    # -----------------------------------
    entries = sorted( glob(f'{result_directory_name}/Scenario*.csv') )

    # plot loss curve
    fig, ax = plt.subplots()
    line_style = ['k^-', 'k^--', 'rs-', 'rs--', 'bo-', 'bo--', 'gd-', 'gd--']
    
    counter = 0
    for (idx, entry) in enumerate(entries):
        print(idx)

        filename_decoded = entry.split("/")[-1].split("_")
        if len(filename_decoded) < 4:
            print("Error in filename")
            continue
        
        if 'enc' not in entry:          
            dataset = filename_decoded[2]
            clustering_method = name_dict[filename_decoded[3].split(".")[0]]
        else:
            dataset = filename_decoded[2]
            clustering_method = name_dict[filename_decoded[3] + '-' + filename_decoded[4].split(".")[0]]
    
        print(f'clustering method is {clustering_method}')

        try:
            df = pd.read_csv(f'{entry}')
        except:
            print("Error reading file")
            continue

        
        markers_on = list(np.arange(0, df.shape[0], marker_step))

        ax.plot(range(len(df['training_accuracy'])), df['training_accuracy'], line_style[2*counter], 
                label=f'{clustering_method}: (train)', linewidth =plot_linewidth, 
                markevery=markers_on, markerfacecolor='none', markersize = marker_size)
        ax.plot(range(len(df['test_accuracy'])), df['test_accuracy'], line_style[2*counter+1], 
                label=f'{clustering_method}: (test)', linewidth =plot_linewidth, 
                markevery=markers_on, markerfacecolor='none', markersize = marker_size)

        counter += 1

    plt.rcParams.update({'font.size': text_size})

    # legend = ax.legend(loc='upper ri')#, shadow=True, fontsize='x-large')
    leg = plt.legend(loc=2, prop={'size': 8})
    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(legend_linewidth)
    # get label texts inside legend and set font size
    # for text in leg.get_texts():
    #     text.set_fontsize(legend_text_size)
    plt.grid(color='k', linestyle=':', linewidth=1, axis='y')
    ax.set_yticks(grid_ticks)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.savefig(f'{result_directory_name}/result.png')
    #plt.show()


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # ----------------------------------
    plt.close('all')
    result_directory_name = f'{args.results_root_dir}/main_fed/'
    folder_list = sorted( glob(f'{result_directory_name}/*/*/') )
    
    for folder in folder_list:
        visualize(folder)
    # ----------------------------------
