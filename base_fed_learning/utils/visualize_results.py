'''
    Visualizes the accuracy results for different methods
    By: Mohammad Abdizadeh & Hadi Jamali-Rad
    e-mails:{moh.abdizadeh, h.jamali.rad}@gmail.com
'''
import matplotlib
# matplotlib.use('Agg')
import sys
sys.path.append("./../")
sys.path.append("./../../")
sys.path.append("./")

import os
import matplotlib.pyplot as plt
from base_fed_learning.utils.options import args_parser
import argparse
from torchvision import datasets, transforms
import torch
import torchvision
import numpy as np
import pandas as pd
from glob import glob

# plotting settings
plot_linewidth = 1.5
text_size = 12
marker_step = 5
marker_size = 7
legend_linewidth = 1.5
legened_location = 4
legend_prop_size = 10
grid_ticks = np.arange(0, 101, 10)

name_dict = {
    'fedavg': 'FedAvg',
    'local': 'Local',
    'fedsem': 'FedSEM',
    'ucfl_enc1': 'Ours (Enc1)',
    'ucfl_enc2': 'Ours (Enc2)'
}

line_style = ['k^-', 'k^--', 'rs-', 'rs--', 'bo-', 'bo--', 'gd-', 'gd--', 'mv-', 'mv--']

def visualize(result_directory_name, include_train =True):
    # -----------------------------------
    entries = sorted( glob(f'{result_directory_name}/Scenario*.csv') )

    # plot loss curve
    fig, ax = plt.subplots()
    
        
    for (idx, entry) in enumerate(entries):
        print(idx)

        filename_decoded = entry.split("/")[-1].split("_")
        if len(filename_decoded) < 4:
            print("Error in filename")
            continue

        # skipping all models_log files
        if filename_decoded[-1] == 'allmodels_log.csv':
            print("skiping log file")
            continue
        
        if 'enc' not in entry:          
            dataset = filename_decoded[2]
            clustering_method = name_dict[filename_decoded[3].split(".")[0]]
        else:
            dataset = filename_decoded[2]
            clustering_method = name_dict[filename_decoded[3] + '_' + filename_decoded[4].split(".")[0]]
    
        print(f'clustering method is {clustering_method}')

        try:
            df = pd.read_csv(f'{entry}')
        except:
            print("Error reading file")
            continue

        markers_on = list(np.arange(0, df.shape[0], marker_step))

        if include_train:
            ax.plot(range(len(df['training_accuracy'])), df['training_accuracy'], line_style[2*idx + 1], 
                    label=f'{clustering_method}: (train)', linewidth =plot_linewidth, 
                    markevery=markers_on, markerfacecolor='none', markersize = marker_size)

        ax.plot(range(len(df['test_accuracy'])), df['test_accuracy'], line_style[2*idx], 
                label=f'{clustering_method}: (test)', linewidth =plot_linewidth, 
                markevery=markers_on, markerfacecolor='none', markersize = marker_size)
        
    # plt.rcParams.update({'font.size': text_size})
    
    plt.rc('font', size=text_size)          # controls default text sizes
    plt.rc('axes', titlesize=text_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=text_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=text_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=text_size)    # fontsize of the tick labels

    plt.rcParams.update({'font.size': text_size})

    # legend = ax.legend(loc='upper ri')#, shadow=True, fontsize='x-large')
    leg = plt.legend(loc=legened_location, prop={'size': legend_prop_size})
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

def visualize_scenario_3(result_array, clustering_methods, result_directory_name):
    # -----------------------------------
    # plot loss curve
    fig, ax = plt.subplots()
    marker_step = 1
    
    for idx in range(len(result_array)):
        print(idx)

        test_data = result_array[idx]

        clustering_method = name_dict[clustering_methods[idx]]
        
        markers_on = list(np.arange(0, test_data.shape[0], marker_step))

        ax.plot(np.arange(len(test_data))*20, test_data, line_style[2*idx], 
                label=f'{clustering_method}', linewidth =plot_linewidth, 
                markevery=markers_on, markerfacecolor='none', markersize = marker_size)

    plt.gca().invert_xaxis()

    # legend = ax.legend(loc='upper ri')#, shadow=True, fontsize='x-large')
    leg = plt.legend(loc=legened_location, prop={'size': legend_prop_size})
    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(legend_linewidth)
    # get label texts inside legend and set font size
    plt.grid(color='k', linestyle=':', linewidth=1, axis='y')
    ax.set_yticks(grid_ticks)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Communication round')
    plt.savefig(f'../{result_directory_name}/result.png')
    plt.show()
    plt.close()

def extract_scenario_3(file_list):
    output_train = []
    output_test = []

    for entry in file_list:
        #print(entry)
        try:
            df = pd.read_csv(f'{entry}')
            #print(df['training_accuracy'].iloc[-1])
            #print(df['test_accuracy'].iloc[-1])
            output_train = output_train + [df['training_accuracy'].iloc[-1]]
            output_test = output_test + [df['test_accuracy'].iloc[-1]]
            
        except:
            print("Error reading file")
            continue
        #visualize(folder)

    return output_train, output_test

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # ----------------------------------
    plt.close('all')
    result_directory_name = f'{args.results_root_dir}/main_fed/'
    
    folder_list = sorted( glob(f'{result_directory_name}/scenario_1/*/') )
    
    for folder in folder_list:
        visualize(folder)
    

    folder_list = sorted( glob(f'{result_directory_name}/scenario_2/*/') )
    
    for folder in folder_list:
        visualize(folder)

    
    folder_list = sorted( glob(f'{result_directory_name}/scenario_4/*/') )
    
    for folder in folder_list:
        visualize(folder)

    
    clustering_methods = ['fedavg', 'local', 'fedsem', 'ucfl_enc1', 'ucfl_enc2']
    result_array = np.empty((0, 6))
    for i, string in enumerate(clustering_methods):
        file_list = sorted( glob(f'../{result_directory_name}scenario_3/CIFAR10/Scenario3_{i+1}*_{string}.csv') )
        output_train, output_test = extract_scenario_3(file_list)
        result_array = np.vstack((result_array, output_test))
    
    visualize_scenario_3(result_array, clustering_methods, f'{result_directory_name}scenario_3/CIFAR10')

    clustering_methods = ['fedavg', 'local', 'fedsem', 'ucfl_enc1', 'ucfl_enc2']
    result_array = np.empty((0, 6))
    for i, string in enumerate(clustering_methods):
        file_list = sorted( glob(f'../{result_directory_name}scenario_3/MNIST/Scenario3_{i+1}*_{string}.csv') )
        output_train, output_test = extract_scenario3(file_list)
        result_array = np.vstack((result_array, output_test))
    
    print(result_array)
    visualize_scenario_3(result_array, clustering_methods, f'{result_directory_name}scenario_3/MNIST')
    # ----------------------------------
