import numpy as np
import matplotlib.pyplot as plt

import time

def plot_GPU_mem_used( batch_sizes, optimizing_batches, gpu_mem_usage):
    
    print('hi')
    
    x = batch_sizes
    
    gpu_mem_usage_mean = gpu_mem_usage.mean(axis=1)
    
    x_lines = []
    y_lines = []
    
    list_ind = 0
    for i in range( len(batch_sizes)):
        x_lines.append( batch_sizes[ : i+1 ] )
        
        y_lines.append( gpu_mem_usage_mean[ list_ind : list_ind + len( optimizing_batches[i] ) ] )
        list_ind = list_ind + len( optimizing_batches[i] )
    for x_line, y_line in zip(x_lines, y_lines):
        print(x_line)
        print(y_line)
        plt.plot(x_line, y_line, marker='o')
    plt.grid(True)
    
def plot_train_time( batch_sizes, optimizing_batches, train_time):
    
    x = np.arange(len(batch_sizes), 0, -1)
    
    # x = batch_sizes    
    train_time_mean = train_time.mean(axis=1)
    
    x_lines = []
    y_lines = []
    legend_names = []
    
    list_ind = 0
    for i in range( len(batch_sizes)):
        x_lines.append( x[ : i+1 ] )        
        y_lines.append( train_time_mean[ list_ind : list_ind + len( optimizing_batches[i] ) ] )
        list_ind = list_ind + len( optimizing_batches[i] )        
        legend_names.append( str(batch_sizes[i]) )
    
    for x_line, y_line, legend_name in zip(x_lines, y_lines, legend_names):
        plt.plot(x_line, y_line, label=legend_name, marker='o')
        
    plt.grid(True)
    leg = plt.legend(title='Backprop Batch')
    leg.set_alpha(0)
    plt.ylabel('Average Training Time(sec)')
    plt.xlabel('Optimizing Batch')
    
    plt.xticks( x, [str(batch_size) for batch_size in batch_sizes] )

if __name__ == '__main__':
    while True:
        time.sleep(1)