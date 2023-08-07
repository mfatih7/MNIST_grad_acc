import numpy as np
import matplotlib.pyplot as plt
import os

import time

def plot_GPU_mem_used( output_path, input_size, input_expand_ratios, batch_sizes, optimizing_batches, optimizer_types, gpu_mem_usage, device_name, device_mem_cap):
    
    gpu_mem_usage_mean_of_epochs = gpu_mem_usage.mean(axis=1)
    
    x = np.arange(len(batch_sizes), 0, -1)
    
    for optimizer_type in optimizer_types:
    
        plt.figure()
    
        input_ress = [str(input_size*input_expand_ratio) for input_expand_ratio in input_expand_ratios]
                    
        x_lines = []
        y_lines = []
        legend_names = []
        
        list_ind = 0
        for i in range( len(input_expand_ratios) ):
            x_lines.append( x )        
            y_line = []
            for j in range( len(batch_sizes) ):
                y_line.append( sum( gpu_mem_usage_mean_of_epochs[ list_ind : list_ind+len(optimizing_batches[j]) ] ) / len( optimizing_batches[j] ) )
                list_ind = list_ind + len( optimizing_batches[j] )
            y_lines.append( y_line )        
            legend_names.append( str(input_ress[i]) + 'x' + str(input_ress[i]) )
        
        for x_line, y_line, legend_name in zip(x_lines, y_lines, legend_names):
            plt.plot(x_line, y_line, label=legend_name, marker='o')
            
        plt.grid(True)
        plt.legend(title='Inpu Size', fontsize = 'small')
        plt.ylabel('Average GPU Memory Usage')
        plt.xlabel('Batch')
        plt.title( device_name + ' ' + str(device_mem_cap) + ' MB ' + optimizer_type + ' opt' )
        
        plt.xticks( x, [str(batch_size) for batch_size in batch_sizes] )
        
        plot_file_name = 'gpu_mem_usage' + optimizer_type + '.png'
        folder_path = os.path.join(output_path, device_name)
        if not os.path.exists( folder_path ):
            os.makedirs(folder_path)            
        plt.savefig( os.path.join(folder_path, plot_file_name), dpi=300)
    
def plot_train_time( output_path, input_size, input_expand_ratios, batch_sizes, optimizing_batches, optimizer_types, train_time, device_name, device_mem_cap):
    
    train_time_mean = train_time.mean(axis=1)
    
    x = np.arange(len(batch_sizes), 0, -1)
    list_ind = 0
    for input_expand_ratio in input_expand_ratios:
        for optimizer_type in optimizer_types:
            plt.figure()
            
            input_res = input_size * input_expand_ratio
                
            x_lines = []
            y_lines = []
            legend_names = []            
            
            for i in range( len(batch_sizes)):
                x_lines.append( x[ : i+1 ] )        
                y_lines.append( train_time_mean[ list_ind : list_ind + len( optimizing_batches[i] ) ] )
                list_ind = list_ind + len( optimizing_batches[i] )        
                legend_names.append( str(batch_sizes[i]) )
            
            for x_line, y_line, legend_name in zip(x_lines, y_lines, legend_names):
                plt.plot(x_line, y_line, label=legend_name, marker='o')
                
            plt.grid(True)
            plt.legend(title='Batch', fontsize = 'small')
            plt.ylabel('Average Training Time(sec)')
            plt.xlabel('Optimizing Batch')
            plt.title( device_name + ' ' + str(device_mem_cap) + ' MB ' + str(input_res) + 'x' + str(input_res) + ' in ' + optimizer_type + ' opt' )
            
            plt.xticks( x, [str(batch_size) for batch_size in batch_sizes] )
            
            plot_file_name = 'gpu_mem_usage' + '_' + str(input_res) + 'x' + str(input_res) + '_' + optimizer_type + '.png'
            folder_path = os.path.join(output_path, device_name)
            if not os.path.exists( folder_path ):
                os.makedirs(folder_path)            
            plt.savefig( os.path.join(folder_path, plot_file_name), dpi=300)

if __name__ == '__main__':
    while True:
        time.sleep(1)