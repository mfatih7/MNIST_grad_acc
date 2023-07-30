import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

from models.models import get_model
from utils import get_gpu_utilization
from utils import get_mem_info


def train_and_test( data_path, learning_rate, n_epochs, num_workers, batch_sizes, optimizing_batches, optimizer_types ):

    transform = transforms.Compose([
        transforms.ToTensor(),           # Convert PIL images to tensors
        transforms.Normalize((0.5,), (0.5,))   # Normalize the image data to the range [-1, 1]
    ])
    
    train_dataset = torchvision.datasets.MNIST(    root = data_path,
                                                   train = True,
                                                   download = True, 
                                                   transform = transform )
    
    
    test_dataset = torchvision.datasets.MNIST(     root = data_path,
                                                   train = False,
                                                   download = True, 
                                                   transform = transform )
    tot_iter = 0
    for batch_size_idx, batch_size in enumerate(batch_sizes):
        tot_iter += len(optimizing_batches[batch_size_idx] * len(optimizer_types) )
    
    train_acc_vec = np.zeros( ( tot_iter, n_epochs ) ) 
    test_acc_vec = np.zeros( ( tot_iter, n_epochs ) ) 
    train_time_vec = np.zeros( ( tot_iter, n_epochs ) ) 
    test_time_vec = np.zeros( ( tot_iter, n_epochs ) ) 
    
    iter_index = 0
    for batch_size_idx, batch_size in enumerate(batch_sizes):
        for optimizing_batch in optimizing_batches[batch_size_idx]:
            for optimizer_type in optimizer_types:
    
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = get_model().to(device)
                criterion = nn.CrossEntropyLoss()            
    
                if(optimizer_type == 'Adam'):
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                
                # Training loop
                print("-" * 80)
                print("Started Training and Test " + str(iter_index) + '/' + str(tot_iter-1) + " for batch size " + str(batch_size) +
                      ", optimizing batch " + str(optimizing_batch) + ", optimizer type " + optimizer_type )            
                
                for epoch in range(n_epochs):
                    
                    correct_train = 0
                    total_train = 0
                    
                    correct_test = 0
                    total_test = 0
                    
                    torch.cuda.reset_peak_memory_stats()
                    
                    start_time_train = time.perf_counter()
                    train_dataloader = DataLoader(  dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    pin_memory = True,
                                                    num_workers = num_workers ) 
                    
                    for i, (images, labels) in enumerate(train_dataloader):
                        images, labels = images.to(device), labels.to(device)
                
                        # Forward pass
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                
                        # Backward and optimize                    
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total_train += labels.size(0)
                        correct_train += (predicted == labels).sum().cpu().detach().numpy()
                        acc_train = correct_train / total_train
                        
                        if( i == 1):
                            print("Train Epoch {}/{} Batch {}/{} LR {:.6f} Loss {:.6f} CorPred {}/{} Acc {:.6f}"
                                    .format(    epoch,
                                                n_epochs-1,
                                                i,
                                                len(train_dataloader)-1,
                                                learning_rate,
                                                loss.cpu().detach().numpy(),
                                                correct_train,
                                                total_train,
                                                acc_train) )
                            
                            print("GPU util: {}" .format( get_gpu_utilization( device, 'cur') ) )
                            print("GPU util: {}" .format( get_gpu_utilization( device, 'max') ) )
                            a,b = get_mem_info( device )
                            print("GPU util: {}/{}" .format( a, b ) )   
                        
                    print("Train Epoch {}/{} Batch {}/{} LR {:.6f} Loss {:.6f} CorPred {}/{} Acc {:.6f}"
                            .format(    epoch,
                                        n_epochs-1,
                                        i,
                                        len(train_dataloader)-1,
                                        learning_rate,
                                        loss.cpu().detach().numpy(),
                                        correct_train,
                                        total_train,
                                        acc_train) )
                    
                    print("GPU util: {}" .format( get_gpu_utilization( device, 'cur') ) )
                    print("GPU util: {}" .format( get_gpu_utilization( device, 'max') ) )
                    a,b = get_mem_info( device )
                    print("GPU util: {}/{}" .format( a, b ) )           
                    
                    # print( torch.cuda.memory_summary() )
                    
                    
                    train_acc_vec[iter_index, epoch] = acc_train
                    end_time_train = time.perf_counter()
                    train_time_vec[iter_index, epoch] = end_time_train - start_time_train
                    
                    torch.cuda.reset_peak_memory_stats()
                    
                    start_time_test = time.perf_counter()
                    test_dataloader = DataLoader( dataset = test_dataset,
                                                  batch_size = batch_size,
                                                  shuffle = False,
                                                  pin_memory = True,
                                                  num_workers = num_workers )
    
                    model.eval()  # Sets the model to evaluation mode
                    with torch.no_grad():        
                        for i, (images, labels) in enumerate(test_dataloader):
                            images, labels = images.to(device), labels.to(device)       
                            
                            outputs = model(images)        
                            
                            _, predicted = torch.max(outputs.data, 1)
                            total_test += labels.size(0)
                            correct_test += (predicted == labels).sum().cpu().detach().numpy()
                            acc_test = correct_test / total_test
                            
                            if( i==1 ):
                                print("Test Epoch {}/{} Batch {}/{} LR {:.6f} CorPred {}/{} Acc {:.6f}"
                                        .format(    epoch,
                                                    n_epochs-1,
                                                    i,
                                                    len(test_dataloader)-1,
                                                    learning_rate,
                                                    correct_test,
                                                    total_test,
                                                    acc_test) )
                                
                                print("GPU util: {}" .format( get_gpu_utilization( device, 'cur') ) )
                                print("GPU util: {}" .format( get_gpu_utilization( device, 'max') ) )
                                a,b = get_mem_info( device )
                                print("GPU util: {}/{}" .format( a, b ) )    
                            
                    print("Test Epoch {}/{} Batch {}/{} LR {:.6f} CorPred {}/{} Acc {:.6f}"
                            .format(    epoch,
                                        n_epochs-1,
                                        i,
                                        len(test_dataloader)-1,
                                        learning_rate,
                                        correct_test,
                                        total_test,
                                        acc_test) )
                    
                    print("GPU util: {}" .format( get_gpu_utilization( device, 'cur') ) )
                    print("GPU util: {}" .format( get_gpu_utilization( device, 'max') ) )
                    a,b = get_mem_info( device )
                    print("GPU util: {}/{}" .format( a, b ) )        
                    
                    # print( torch.cuda.memory_summary() )
                                
                    test_acc_vec[iter_index, epoch] = acc_test
                    end_time_test = time.perf_counter()
                    test_time_vec[iter_index, epoch] = end_time_test - start_time_test
                            
                    print("-" * 40)
                iter_index = iter_index + 1

    
    
    