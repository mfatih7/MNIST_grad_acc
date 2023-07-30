# import MNIST_train_test_grad_acc
import MNIST_train_test_grad_acc_2


data_path = './'

# Hyperparameters
learning_rate = 0.001
n_epochs = 1
num_workers = 0


batch_sizes = [  256,
                 128, 
                  64,
                  32,
                  16,
                   8,
                   4,  ]

optimizing_batches = [  [  2**0,  ], 
                        [  2**0,  2**1,  ],
                        [  2**0,  2**1,  2**2,  ], 
                        [  2**0,  2**1,  2**2,  2**3,  ], 
                        [  2**0,  2**1,  2**2,  2**3,  2**4,  ], 
                        [  2**0,  2**1,  2**2,  2**3,  2**4,  2**5,  ],
                        [  2**0,  2**1,  2**2,  2**3,  2**4,  2**5,  2**6,  ],  ]

# optimizer_types = ['SGD', 'ADAM']
optimizer_types = ['ADAM']


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