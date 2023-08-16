import numpy as np
import matplotlib.pyplot as plt
import os

import time

def plot_GPU_mem_used( output_path, input_size, training_params, optimizer_types, gpu_mem_usage, device_name, device_mem_cap):
    
    gpu_mem_usage_max_of_epochs = gpu_mem_usage.max(axis=1)    
    gpu_mem_usage_max_of_epochs = gpu_mem_usage_max_of_epochs.reshape(-1, len(optimizer_types) )
    
    x_ticks = []    
    for training_param in training_params:
        x_ticks.append(training_param[1])
    x_ticks = sorted( list( set(x_ticks) ) )
    
    for optimizer_type_id, optimizer_type in enumerate(optimizer_types):
    
        plt.figure()                           
        x_lines = []
        y_lines = []
        legend_names = []
        start_ind = 0
        list_ind = 0        
        
        while ( list_ind < gpu_mem_usage_max_of_epochs.shape[0] ):
            x_line = []
            y_line = []
            start_ind = list_ind
            cur_exp_rat = training_params[list_ind][0]
            cur_batch = training_params[list_ind][1]
            list_ind = list_ind + 1
            while ( list_ind < gpu_mem_usage_max_of_epochs.shape[0] ):
                if(cur_exp_rat == training_params[list_ind][0] and cur_batch == training_params[list_ind][1]):
                    list_ind = list_ind + 1
                elif(cur_exp_rat == training_params[list_ind][0] and cur_batch != training_params[list_ind][1]):
                    y_line.append( max(gpu_mem_usage_max_of_epochs[ start_ind:list_ind, optimizer_type_id]) )
                    x_line.append( x_ticks.index(cur_batch) )
                    start_ind = list_ind
                    cur_exp_rat = training_params[list_ind][0]
                    cur_batch = training_params[list_ind][1]
                    list_ind = list_ind + 1
                else:
                    break
                
            y_line.append( max(gpu_mem_usage_max_of_epochs[ start_ind:list_ind, optimizer_type_id]) )
            x_line.append( x_ticks.index(cur_batch) )
            
            y_lines.append(y_line)
            x_lines.append(x_line)
            legend_names.append( str(input_size*cur_exp_rat) + 'x' + str(input_size*cur_exp_rat) )
            
        for x_line, y_line, legend_name in zip(x_lines, y_lines, legend_names):
            plt.plot(x_line, y_line, label=legend_name, marker='o')
            
        plt.grid(True)
        plt.legend(title='Input Size', fontsize = 'small')
        plt.ylabel('Max GPU Memory Usage')
        plt.xlabel('Batch Size')
        plt.title( device_name + ' ' + str(device_mem_cap) + ' MB_' + optimizer_type + ' opt' )
        
        plt.xticks( range(len(x_ticks)), [str(x_tick) for x_tick in x_ticks] )
        
        plot_file_name = 'gpu_mem_usage' + optimizer_type + '.png'
        folder_path = os.path.join(output_path, device_name)
        if not os.path.exists( folder_path ):
            os.makedirs(folder_path)            
        plt.savefig( os.path.join(folder_path, plot_file_name), dpi=300)            

def plot_GPU_util( output_path, input_size, training_params, optimizer_types, gpu_util, device_name, device_mem_cap):
    
    gpu_util_mean_of_epochs = gpu_util.mean(axis=1)    
    gpu_util_mean_of_epochs = gpu_util_mean_of_epochs.reshape(-1, len(optimizer_types) )
    
    x_ticks = []    
    for training_param in training_params:
        x_ticks.append(training_param[1])
    x_ticks = sorted( list( set(x_ticks) ) )
    
    for optimizer_type_id, optimizer_type in enumerate(optimizer_types):
    
        plt.figure()                           
        x_lines = []
        y_lines = []
        legend_names = []
        start_ind = 0
        list_ind = 0        
        
        while ( list_ind < gpu_util_mean_of_epochs.shape[0] ):
            x_line = []
            y_line = []
            start_ind = list_ind
            cur_exp_rat = training_params[list_ind][0]
            cur_batch = training_params[list_ind][1]
            list_ind = list_ind + 1
            while ( list_ind < gpu_util_mean_of_epochs.shape[0] ):
                if(cur_exp_rat == training_params[list_ind][0] and cur_batch == training_params[list_ind][1]):
                    list_ind = list_ind + 1
                elif(cur_exp_rat == training_params[list_ind][0] and cur_batch != training_params[list_ind][1]):
                    y_line.append( max(gpu_util_mean_of_epochs[ start_ind:list_ind, optimizer_type_id]) )
                    x_line.append( x_ticks.index(cur_batch) )
                    start_ind = list_ind
                    cur_exp_rat = training_params[list_ind][0]
                    cur_batch = training_params[list_ind][1]
                    list_ind = list_ind + 1
                else:
                    break
                
            y_line.append( max(gpu_util_mean_of_epochs[ start_ind:list_ind, optimizer_type_id]) )
            x_line.append( x_ticks.index(cur_batch) )
            
            y_lines.append(y_line)
            x_lines.append(x_line)
            legend_names.append( str(input_size*cur_exp_rat) + 'x' + str(input_size*cur_exp_rat) )
            
        for x_line, y_line, legend_name in zip(x_lines, y_lines, legend_names):
            plt.plot(x_line, y_line, label=legend_name, marker='o')
            
        plt.grid(True)
        plt.legend(title='Input Size', fontsize = 'small')
        plt.ylabel('Average GPU Util %')
        plt.xlabel('Batch Size')
        plt.title( device_name + ' ' + str(device_mem_cap) + ' MB_' + optimizer_type + ' opt' )
        
        plt.xticks( range(len(x_ticks)), [str(x_tick) for x_tick in x_ticks] )
        
        plot_file_name = 'gpu_util' + optimizer_type + '.png'
        folder_path = os.path.join(output_path, device_name)
        if not os.path.exists( folder_path ):
            os.makedirs(folder_path)            
        plt.savefig( os.path.join(folder_path, plot_file_name), dpi=300)    

def plot_train_time( output_path, input_size, training_params, optimizer_types, train_time, device_name, device_mem_cap):
    
    train_time_mean = train_time.mean(axis=1)
    
    train_time_mean = train_time_mean.reshape(-1, len(optimizer_types) )
    
    input_expand_ratios = []
    batch_sizes = []
    optimizing_batches = []
    for training_param in training_params:
        input_expand_ratios.append(training_param[0])
        batch_sizes.append(training_param[1])
        optimizing_batches.append(training_param[2])
    input_expand_ratios = sorted( list( set(input_expand_ratios) ) )
    batch_sizes = sorted( list( set(batch_sizes) ) )
    optimizing_batches = sorted( list( set(optimizing_batches) ) )
    
    for optimizer_type_id, optimizer_type in enumerate(optimizer_types):
        list_ind = 0
        for input_expand_ratio in reversed(input_expand_ratios):
            plt.figure()
            
            input_res = input_size * input_expand_ratio
            
            x_lines = []
            y_lines = []
            legend_names = []            
            
            while( list_ind < train_time_mean.shape[0] and training_params[list_ind][0] == input_expand_ratio):
                
                x_line = []
                y_line = []
                
                cur_batch = training_params[list_ind][1]
                
                while( list_ind < train_time_mean.shape[0] and training_params[list_ind][1] == cur_batch):
                    x_line.append( optimizing_batches.index(training_params[list_ind][2]) )
                    y_line.append( train_time_mean[list_ind,optimizer_type_id] )
                    list_ind = list_ind + 1
                
                x_lines.append(x_line)
                y_lines.append(y_line)
                legend_names.append(str(cur_batch))
                
            for x_line, y_line, legend_name in zip(x_lines, y_lines, legend_names):
                plt.plot(x_line, y_line, label=legend_name, marker='o')
                
            plt.grid(True)
            plt.legend(title='Batch', fontsize = 'small')
            plt.ylabel('Average Training Time/Epoch (sec)')
            plt.xlabel('Optimizing Batch')
            plt.title( device_name + ' ' + str(device_mem_cap) + ' MB ' + str(input_res) + 'x' + str(input_res) + ' in ' + optimizer_type + ' opt' )
            
            plt.xticks( range(len(optimizing_batches)), [str(optimizing_batch) for optimizing_batch in optimizing_batches] )
            
            plot_file_name = 'training_time' + '_' + str(input_res) + 'x' + str(input_res) + '_' + optimizer_type + '.png'
            folder_path = os.path.join(output_path, device_name)
            if not os.path.exists( folder_path ):
                os.makedirs(folder_path)            
            plt.savefig( os.path.join(folder_path, plot_file_name), dpi=300)

if __name__ == '__main__':
    while True:
        time.sleep(1)