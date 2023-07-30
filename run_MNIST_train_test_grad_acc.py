import MNIST_train_test_grad_acc

data_path = './'

# Hyperparameters
learning_rate = 0.001
n_epochs = 3
num_workers = 0

# batch_sizes = [ 128, 128, 128, 128, 128, 128, 128, 128, 
#                   64,  64,  64,  64,  64,  64,  64, 
#                   32,  32,  32,  32,  32,  32,
#                   16,  16,  16,  16,  16, ]
# optimizing_batches = [    [ 128, 64, 32, 16,  8,  4,  2,  1, ],
#                           [ 64,  32, 16,  8,  4,  2,  1, ],
#                           [ 32,  16,  8,  4,  2,  1, ],
#                           [ 16,   8,  4,  2,  1 ], ]
# optimizer_types = ['SGD', 'Adam']

batch_sizes = [  128,
                 64,
                 32,
                 16,
                 8,  ]

optimizing_batches = [  [ 128, ], 
                        [ 64, ], 
                        [ 32, ], 
                        [ 16, ], 
                        [ 8, ], ]
optimizer_types = ['SGD']


if __name__ == '__main__':
    MNIST_train_test_grad_acc.train_and_test(   data_path,
                                                learning_rate,
                                                n_epochs,
                                                num_workers,
                                                batch_sizes,
                                                optimizing_batches,
                                                optimizer_types)