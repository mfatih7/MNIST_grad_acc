# import MNIST_train_test_grad_acc
import MNIST_train_test_grad_acc_3

import threading
import queue

from utils import get_info
from plots import plot_GPU_mem_used 
from plots import plot_train_time 

data_path = './'
input_size = 28
bn_or_gn = 'bn'
bn_or_gn = 'gn'

# Hyperparameters
learning_rate = 0.001
n_epochs = 2
num_workers = 1

# input_expand_ratios = [ 1, 2, 4, 8, 16, 32, 64 ]
input_expand_ratios = [1, 2, 4] # 1, 2, 4, 8, 16, 32, 64

# batch_sizes = [  256,
#                  128, 
#                   64,
#                   32,
#                   16,
#                    8,
#                    4,  ]

# optimizing_batches = [  [  2**0,  ], 
#                         [  2**0,  2**1,  ],
#                         [  2**0,  2**1,  2**2,  ], 
#                         [  2**0,  2**1,  2**2,  2**3,  ], 
#                         [  2**0,  2**1,  2**2,  2**3,  2**4,  ], 
#                         [  2**0,  2**1,  2**2,  2**3,  2**4,  2**5,  ],
#                         [  2**0,  2**1,  2**2,  2**3,  2**4,  2**5,  2**6,  ],  ]

batch_sizes = [  128,
                 16,
                 2,  ]

optimizing_batches = [  [  2**0,  ], 
                        [  2**0,  2**3,  ],
                        [  2**0,  2**3,  2**6,  ],  ]

# optimizer_types = ['SGD', 'ADAM']
optimizer_types = ['ADAM']

    
if __name__ == '__main__':
    
    event_start_read_GPU_info = threading.Event()
    queue_gpu_info = queue.Queue()
    queue_training_results = queue.Queue()
    
    gpu_thread = threading.Thread( target = get_info, daemon=True, args=(event_start_read_GPU_info, queue_gpu_info,) ) 
    training_thread = threading.Thread( target = MNIST_train_test_grad_acc_3.train_and_test, args=(
                                                                                                        data_path,
                                                                                                        input_size,
                                                                                                        input_expand_ratios,
                                                                                                        learning_rate,
                                                                                                        n_epochs,
                                                                                                        num_workers,
                                                                                                        batch_sizes,
                                                                                                        bn_or_gn,
                                                                                                        optimizing_batches,
                                                                                                        optimizer_types,
                                                                                                        event_start_read_GPU_info,
                                                                                                        queue_gpu_info,
                                                                                                        queue_training_results, ) )

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
    
    plot_GPU_mem_used( input_size, input_expand_ratios, batch_sizes, optimizing_batches, optimizer_types, train_gpu_mem_usage, device_name, device_mem_cap)
    plot_train_time( input_size, input_expand_ratios, batch_sizes, optimizing_batches, optimizer_types, train_time_vec, device_name, device_mem_cap)
        
    