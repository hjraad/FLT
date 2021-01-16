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

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # ----------------------------------
    plt.close('all')

    # ----------------------------------
    # plotting settings
    plot_linewidth = 3
    text_size = 5
    marker_step = 5
    marker_size = 5
    legend_text_size = 'x-large'
    legend_linewidth = 4

    # -----------------------------------

    result_directory_name = f'{args.results_root_dir}/main_fed/'
    folder_list = sorted( glob(f'{result_directory_name}/scenario_*/') )
    print(folder_list)
    for folder in folder_list:
        results_path = folder
        entries = sorted( glob(f'{results_path}/Scenario*.csv') )
    
        # plot loss curve
        fig, ax = plt.subplots()
        line_style = ['k^-', 'k-.', 'rs-', 'r-.', 'bo-', 'b-.', 'gd-', 'g-.']
        
        counter = 0
        for (idx, entry) in enumerate(entries):
            print(idx)

            filename_decoded = entry.split("/")[-1].split("_")
            if len(filename_decoded) < 3:
                print("Error in filename")
                continue

            dataset = filename_decoded[1]
            clustering_method = filename_decoded[2].split(".")[0]
       
            print(f'clustering method is {clustering_method}')

            try:
                df = pd.read_csv(f'{entry}')
            except:
                print("Error reading file")
                continue

            
            markers_on = list(np.arange(0, df.shape[0], marker_step))

            ax.plot(range(len(df['training_accuracy'])), df['training_accuracy'], line_style[2*counter], 
                    label=f'{clustering_method}: train_accuracy', linewidth =plot_linewidth, markevery=markers_on, markersize = marker_size)
            ax.plot(range(len(df['test_accuracy'])), df['test_accuracy'], line_style[2*counter+1], 
                    label=f'{clustering_method}: test_accuracy', linewidth =plot_linewidth, markevery=markers_on, markersize = marker_size)

            counter += 1

        plt.rcParams.update({'font.size': text_size})

        legend = ax.legend(loc='lower right')#, shadow=True, fontsize='x-large')
        leg = plt.legend()
        # get the individual lines inside legend and set line width
        for line in leg.get_lines():
            line.set_linewidth(legend_linewidth)
        # get label texts inside legend and set font size
        for text in leg.get_texts():
            text.set_fontsize(legend_text_size)
        plt.grid(color='k', linestyle=':', linewidth=1, axis='y')
        plt.ylabel('accuracy (%)')
        plt.xlabel('epoch')
        plt.savefig(f'{results_path}/result.png')
        plt.show()
"""
-   accuracy (%)
-    Text size (on axis, labels and titles and legend) encode in
-   dotted line -> dashed
-    thickness line -> encode in the program
-    marker (^ , s, o, d), and its size encoded
-   spacing of markers (every 5)
    Naming convention, and order:
        FedAvg (train),
        FedAvg (test)
        Local (train),
        Local (test),
        FedSem (train),
        FedSem (test),
        Ours (train)
        Ours (test)
-        Axis grid -> only horizontal and every 10%
"""