# import MNIST_train_test_grad_acc
import MNIST_train_test_grad_acc_3

import threading
import queue

from utils import get_info
from plots import plot_GPU_mem_used 
from plots import plot_train_time
from plots import plot_GPU_util 

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

# training_params -> [input_expand_ratios, batch_sizes, optimizing_batches]

# training_params = [ 
                    
#                     [32, 16, 1],
#                     [32,  8, 1],[32,  8, 2],
#                     [32,  4, 1],[32,  4, 2],[32,  4, 4],                 
                    
#                     [16, 32, 1],
#                     [16, 16, 1],[16, 16, 2],
#                     [16,  8, 1],[16,  8, 2],[16,  8, 4],
#                     [16,  4, 1],[16,  4, 2],[16,  4, 4],[16, 4, 8],
                    
#                     [ 8, 64, 1],
#                     [ 8, 32, 1],[ 8, 32, 2],
#                     [ 8, 16, 1],[ 8, 16, 2],[ 8, 16, 4],
#                     [ 8,  8, 1],[ 8,  8, 2],[ 8,  8, 4],[ 8,  8, 8],
#                     [ 8,  4, 1],[ 8,  4, 2],[ 8,  4, 4],[ 8,  4, 8],[ 8, 4, 16],
                    
#                     [ 4,128, 1],
#                     [ 4, 64, 1],[ 4, 64, 2],
#                     [ 4, 32, 1],[ 4, 32, 2],[ 4, 32, 4],
#                     [ 4, 16, 1],[ 4, 16, 2],[ 4, 16, 4],[ 4, 16, 8],
#                     [ 4,  8, 1],[ 4,  8, 2],[ 4,  8, 4],[ 4,  8, 8],[ 4,  8,16],
#                     [ 4,  4, 1],[ 4,  4, 2],[ 4,  4, 4],[ 4,  4, 8],[ 4,  4,16],[ 4,  4, 32],
#                   ]

training_params = [ 
                    [32, 32, 1],
                    [32, 16, 1],[32, 16, 2],
                    [32,  8, 1],[32,  8, 2],[32,  8, 4],
                    [32,  4, 1],[32,  4, 2],[32,  4, 4],[32,  4, 8],              
                    
                    [16, 64, 1],
                    [16, 32, 1],[16, 32, 2],
                    [16, 16, 1],[16, 16, 2],[16, 16, 4],
                    [16,  8, 1],[16,  8, 2],[16,  8, 4],[16,  8, 8],
                    [16,  4, 1],[16,  4, 2],[16,  4, 4],[16,  4, 8],[16,  4, 16],
                    
                    [ 8,128, 1],
                    [ 8, 64, 1],[ 8, 64, 1],
                    [ 8, 32, 1],[ 8, 32, 2],[ 8, 32, 4],
                    [ 8, 16, 1],[ 8, 16, 2],[ 8, 16, 4],[ 8, 16, 8],
                    [ 8,  8, 1],[ 8,  8, 2],[ 8,  8, 4],[ 8,  8, 8],[ 8,  8, 16],
                    [ 8,  4, 1],[ 8,  4, 2],[ 8,  4, 4],[ 8,  4, 8],[ 8,  4, 16],[ 8,  4, 32],
                    
                    [ 4,256, 1],
                    [ 4,128, 1],[ 4,128, 2],
                    [ 4, 64, 1],[ 4, 64, 2],[ 4, 64, 4],
                    [ 4, 32, 1],[ 4, 32, 2],[ 4, 32, 4],[ 4, 32, 8],
                    [ 4, 16, 1],[ 4, 16, 2],[ 4, 16, 4],[ 4, 16, 8],[ 4, 16,16],
                    [ 4,  8, 1],[ 4,  8, 2],[ 4,  8, 4],[ 4,  8, 8],[ 4,  8,16],[ 4  ,8, 32],
                    [ 4,  4, 1],[ 4,  4, 2],[ 4,  4, 4],[ 4,  4, 8],[ 4,  4,16],[ 4,  4, 32],[ 4,  8, 64],
                  ]
    
    
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

    
if __name__ == '__main__':
    
    event_start_read_GPU_info = threading.Event()
    queue_gpu_info = queue.Queue()
    queue_training_results = queue.Queue()
    
    gpu_thread = threading.Thread( target = get_info, daemon=True, args=(event_start_read_GPU_info, queue_gpu_info,) ) 
    training_thread = threading.Thread( target = MNIST_train_test_grad_acc_3.train_and_test, args=(
                                                                                                        data_path,
                                                                                                        input_size,
                                                                                                        learning_rate,
                                                                                                        n_epochs,
                                                                                                        num_workers,
                                                                                                        bn_or_gn,
                                                                                                        en_grad_checkpointing,
                                                                                                        training_params,
                                                                                                        optimizer_types,
                                                                                                        event_start_read_GPU_info,
                                                                                                        queue_gpu_info,
                                                                                                        queue_training_results, ) )

    # MNIST_train_test_grad_acc_3.train_and_test(     data_path,
    #                                                 input_size,
    #                                                 learning_rate,
    #                                                 n_epochs,
    #                                                 num_workers,
    #                                                 bn_or_gn,
    #                                                 en_grad_checkpointing,
    #                                                 training_params,
    #                                                 optimizer_types,
    #                                                 event_start_read_GPU_info,
    #                                                 queue_gpu_info,
    #                                                 queue_training_results, )

    # Start the threads
    gpu_thread.start()
    training_thread.start()

    # Optionally, wait for the threads to complete
    training_thread.join()
    # gpu_thread.join() # no need to join(wait to finish since the thread is daemon)
    
    training_results = queue_training_results.get()

    train_acc_vec       = training_results[0]
    test_acc_vec        = training_results[1]
    train_time_vec      = training_results[2]
    test_time_vec       = training_results[3]
    train_gpu_mem_usage = training_results[4]
    test_gpu_mem_usage  = training_results[5]
    train_gpu_util      = training_results[6]
    test_gpu_util       = training_results[7]
    device_name         = training_results[8]
    device_mem_cap      = training_results[9]
    
    plot_GPU_mem_used( output_path, input_size, training_params, optimizer_types, train_gpu_mem_usage, device_name, device_mem_cap)
    plot_GPU_util( output_path, input_size, training_params, optimizer_types, train_gpu_util, device_name, device_mem_cap)
    plot_train_time( output_path, input_size, training_params, optimizer_types, train_time_vec, device_name, device_mem_cap)

    