import numpy as np

from plots import plot_GPU_mem_used 
from plots import plot_train_time 


data_path = './'
output_path = './outputs'
input_size = 28
bn_or_gn = 'bn'
bn_or_gn = 'gn'
en_grad_checkpointing = False
en_grad_checkpointing = True

# Hyperparameters
learning_rate = 0.001
n_epochs = 2
num_workers = 1    
    
training_params = [
                    [4,128, 1], 
                    [4, 16, 1],[4, 16, 8],
                    [4,  2, 1],[4,  2, 8],[4,  2,16],
                    
                    [2,128, 1], 
                    [2, 16, 1],[2, 16, 8],
                    [2,  2, 1],[2,  2, 8],[2,  2,16],
                    
                    
                  ]

optimizer_types = ['SGD', 'ADAM']
# optimizer_types = ['ADAM']

train_gpu_mem_usage = np.array( [ [101, 102], [103, 104], [105, 106], [107, 108],
                                  [109, 110], [111, 112], [113, 114], [115, 116],
                                  [117, 118], [119, 120], [121, 122], [123, 124],
                                  [125, 126], [127, 128], [129, 130], [131, 132],
                                  [133, 134], [135, 136], [137, 138], [139, 140],
                                  [141, 142], [143, 144], [145, 146], [147, 148],] )

device_name = 'asd'
device_mem_cap = 4000

plot_GPU_mem_used( output_path, input_size, training_params, optimizer_types, train_gpu_mem_usage, device_name, device_mem_cap)
