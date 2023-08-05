import time

import matplotlib.pyplot as plt

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
    
    print('hi')
    
    x = batch_sizes
    
    train_time_mean = train_time.mean(axis=1)
    
    x_lines = []
    y_lines = []
    legend_names = []
    
    list_ind = 0
    for i in range( len(batch_sizes)):
        x_lines.append( batch_sizes[ : i+1 ] )
        
        y_lines.append( train_time_mean[ list_ind : list_ind + len( optimizing_batches[i] ) ] )
        list_ind = list_ind + len( optimizing_batches[i] )
        
        legend_names.append( 'Backprop Batch ' + str(x[i]))
    
    for x_line, y_line, legend_name in zip(x_lines, y_lines, legend_names):
        print(x_line)
        print(y_line)
        plt.plot(x_line, y_line, label=legend_name, marker='o')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Average Training Time')
    plt.xlabel('Optimizing Batch')
            

if __name__ == '__main__':
    while True:
        time.sleep(1)