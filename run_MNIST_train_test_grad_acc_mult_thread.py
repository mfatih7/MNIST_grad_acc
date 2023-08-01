# import MNIST_train_test_grad_acc
import MNIST_train_test_grad_acc_3

import threading
import queue
import time

from utils import return_gpu_info 

data_path = './'

# Hyperparameters
learning_rate = 0.001
n_epochs = 5
num_workers = 1


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

batch_sizes = [  256,
                 128,  ]

optimizing_batches = [  [  2**0,  ], 
                        [  2**0,  2**1,  ],  ]

# optimizer_types = ['SGD', 'ADAM']
optimizer_types = ['ADAM']


def get_info( event_start_read_GPU_info, queue_gpu_info ):    
    while True:
        event_start_read_GPU_info.wait()
        # print('Event cleared')
        event_start_read_GPU_info.clear()
        queue_gpu_info.put(return_gpu_info())      
    
if __name__ == '__main__':
    
    event_start_read_GPU_info = threading.Event()
    queue_gpu_info = queue.Queue()
    queue_training_results = queue.Queue()
    
    gpu_thread = threading.Thread( target = get_info, daemon=True, args=(event_start_read_GPU_info, queue_gpu_info,) ) 
    training_thread = threading.Thread( target = MNIST_train_test_grad_acc_3.train_and_test, args=(
                                                                                                        data_path,
                                                                                                        learning_rate,
                                                                                                        n_epochs,
                                                                                                        num_workers,
                                                                                                        batch_sizes,
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

    train_acc_vec = training_results[0]
    test_acc_vec = training_results[1]
    train_time_vec = training_results[2]
    test_time_vec = training_results[3]