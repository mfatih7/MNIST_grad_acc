# import MNIST_train_test_grad_acc
import MNIST_train_test_grad_acc_3

import threading
import queue
import time

from utils import return_gpu_info 

data_path = './'

# Hyperparameters
learning_rate = 0.001
n_epochs = 10
num_workers = 0


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


def get_info( gpu_info ):    
    while True:
        if gpu_info.empty():   
            gpu_info.put(return_gpu_info())
        time.sleep(0.001)      
    
if __name__ == '__main__':
        
    gpu_info = queue.Queue()
    training_queue = queue.Queue()
    
    gpu_thread = threading.Thread( target = get_info, daemon=True, args=(gpu_info,) ) 
    training_thread = threading.Thread( target = MNIST_train_test_grad_acc_3.train_and_test, args=(
                                                                                                        data_path,
                                                                                                        learning_rate,
                                                                                                        n_epochs,
                                                                                                        num_workers,
                                                                                                        batch_sizes,
                                                                                                        optimizing_batches,
                                                                                                        optimizer_types,
                                                                                                        gpu_info,
                                                                                                        training_queue,) )

    # Start the threads
    gpu_thread.start()
    training_thread.start()

    # Optionally, wait for the threads to complete
    training_thread.join()
    # gpu_thread.join() # no need to join(wait to finish since the thread is daemon)
    
    training_results = training_queue.get()

    train_acc_vec = training_results[0]
    test_acc_vec = training_results[1]
    train_time_vec = training_results[2]
    test_time_vec = training_results[3]