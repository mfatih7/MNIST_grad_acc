import time

import matplotlib.pyplot as plt

def plot_GPU_mem_used( batch_sizes, optimizing_batches, gpu_mem_usage):
    
    print('hi')
    
    x = batch_sizes
    
    x_lines = []
    
    for i in range( len(batch_sizes)):
        print( str(batch_sizes[i]) )
        x_lines.append( batch_sizes[i:] )
    

if __name__ == '__main__':
    while True:
        time.sleep(1)