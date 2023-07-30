# import MNIST_train_test_grad_acc
import MNIST_train_test_grad_acc_2


data_path = './'

# Hyperparameters
learning_rate = 0.001
n_epochs = 3
num_workers = 1


# batch_sizes = [  128, 
#                   64,
#                   32,
#                   16,
#                    8,
#                    4,  ]

# optimizing_batches = [  [  128,  ], 
#                         [  64,  128,  ],
#                         [  32,   64,   128,  ], 
#                         [  16,   32,    64,   128,  ], 
#                         [   8,   16,    32,    64,   128,  ], 
#                         [   4,    8,    16,    32,    64,   128,  ],  ]

# optimizer_types = ['SGD', 'ADAM']

batch_sizes = [  128, 
                  64,
                  32,  ]

optimizing_batches = [  [  128,  ], 
                        [  64,  128,  ],
                        [  32,   64,   128,  ],]

optimizer_types = ['SGD', 'ADAM']


# if __name__ == '__main__':
#     results = MNIST_train_test_grad_acc.train_and_test(  data_path,
#                                                          learning_rate,
#                                                          n_epochs,
#                                                          num_workers,
#                                                          batch_sizes,
#                                                          optimizing_batches,
#                                                          optimizer_types  )    
#     train_acc_vec = results[0]
#     test_acc_vec = results[1]
#     train_time_vec = results[2]
#     test_time_vec = results[3]
    
if __name__ == '__main__':
    results = MNIST_train_test_grad_acc_2.train_and_test(   data_path,
                                                            learning_rate,
                                                            n_epochs,
                                                            num_workers,
                                                            batch_sizes,
                                                            optimizing_batches,
                                                            optimizer_types,  )    
    train_acc_vec = results[0]
    test_acc_vec = results[1]
    train_time_vec = results[2]
    test_time_vec = results[3]