import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import time

from torchsummary import summary

from utils import clear_gpu_mem_util
from utils import check_gpu_info_queue
from utils import trig_GPU_read
from utils import get_info_from_GPU_queue

from models.models import get_model
from models.torchSummaryWrapper import get_torchSummaryWrapper

def train_and_test( data_path,
                    input_size,
                    input_expand_ratios,
                    learning_rate,
                    n_epochs,
                    num_workers,
                    batch_sizes,
                    bn_or_gn,
                    en_grad_checkpointing,
                    optimizing_batches,
                    optimizer_types,
                    event_start_read_GPU_info,
                    queue_gpu_info,
                    queue_training_results, ):
    
    
    tot_iter = 0
    for batch_size_idx, batch_size in enumerate(batch_sizes):
        tot_iter += len(optimizing_batches[batch_size_idx] * len(optimizer_types) )
    tot_iter = tot_iter * len(input_expand_ratios)
    
    train_acc_vec = np.zeros( ( tot_iter, n_epochs ) ) 
    test_acc_vec = np.zeros( ( tot_iter, n_epochs ) ) 
    train_time_vec = np.zeros( ( tot_iter, n_epochs ) ) 
    test_time_vec = np.zeros( ( tot_iter, n_epochs ) ) 
    train_gpu_mem_usage = np.zeros( ( tot_iter, n_epochs ) ) 
    test_gpu_mem_usage = np.zeros( ( tot_iter, n_epochs ) )
    train_gpu_util = np.zeros( ( tot_iter, n_epochs ) ) 
    test_gpu_util = np.zeros( ( tot_iter, n_epochs ) )
    
    iter_index = 0    
    for input_expand_ratio in input_expand_ratios:

        transform = transforms.Compose([
            transforms.Resize( [input_size*input_expand_ratio, input_size*input_expand_ratio ] ),
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

        for batch_size_idx, batch_size in enumerate(batch_sizes):
            for optimizing_batch in optimizing_batches[batch_size_idx]:
                for optimizer_type in optimizer_types:
        
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    if( device == 'gpu' ):
                        torch.cuda.empty_cache()
                    
                    # device = 'cpu'
                    
                    model = get_model( input_expand_ratio, bn_or_gn, en_grad_checkpointing ).to(device)
                    criterion = nn.CrossEntropyLoss()          
                    
                    # if(en_grad_checkpointing==False):
                    #     summary(model, (1, input_size*input_expand_ratio, input_size*input_expand_ratio) )
                    # else:                    
                    #     summary(get_torchSummaryWrapper( model ), (1, input_size*input_expand_ratio, input_size*input_expand_ratio) )                        
        
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
                        
                        gpu_mem_usage_max_train = 0
                        gpu_util_max_train = 0
                        
                        gpu_mem_usage_max_test = 0
                        gpu_util_max_test = 0
                        
                        device_name = []
                        device_mem_cap = 0
                        
                        start_time_train = time.perf_counter()
                        train_dataloader = DataLoader(  dataset = train_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = True,
                                                        pin_memory = True,
                                                        num_workers = num_workers ) 
                        
                        for i, (images, labels) in enumerate(train_dataloader):
                            
                            trig_GPU_read( queue_gpu_info, event_start_read_GPU_info )
                            
                            images.requires_grad = True
                            # print(images.requires_grad)
                            
                            images, labels = images.to(device), labels.to(device)
                            
                            # Forward pass
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            
                            loss = loss / optimizing_batch       
                            
                            # Backward and optimize                    
                            loss.backward()
                            
                            _, predicted = torch.max(outputs.data, 1)
                            total_train += labels.size(0)
                            correct_train += (predicted == labels).sum().cpu().detach().numpy()
                            acc_train = correct_train / total_train
                            
                            if( (i+1) % optimizing_batch == 0 or (i+1) == len(train_dataloader) ):
                                optimizer.step()
                                optimizer.zero_grad()          
                            
                            data_from_GPU_queue = get_info_from_GPU_queue( queue_gpu_info, event_start_read_GPU_info)
                            if(data_from_GPU_queue != None):
                                device_name = data_from_GPU_queue[0]
                                device_mem_cap = data_from_GPU_queue[1]
                                gpu_mem_usage_max_train = max(gpu_mem_usage_max_train, data_from_GPU_queue[2])
                                gpu_util_max_train = max(gpu_util_max_train, data_from_GPU_queue[3])                                                        
                            
                            if( ( (i*batch_size) % 10000 ) > ( ((i+1)*batch_size) % 10000 ) or (i+1) == len(train_dataloader) ):                               
                                
                                print("Train Epoch {}/{} Batch {}/{} LR {:.6f} Loss {:.6f} CorPred {}/{} Acc {:.6f} {} Mem Used {}/{} GPU Util {}"
                                        .format(    epoch,
                                                    n_epochs-1,
                                                    i,
                                                    len(train_dataloader)-1,
                                                    learning_rate,
                                                    loss.cpu().detach().numpy(),
                                                    correct_train,
                                                    total_train,
                                                    acc_train,
                                                    device_name,
                                                    gpu_mem_usage_max_train,
                                                    device_mem_cap,
                                                    gpu_util_max_train, ) )   
                        
                        check_gpu_info_queue(queue_gpu_info)
                        
                        train_acc_vec[iter_index, epoch]        = acc_train
                        end_time_train                          = time.perf_counter()
                        train_time_vec[iter_index, epoch]       = end_time_train - start_time_train
                        train_gpu_mem_usage[iter_index, epoch]  = gpu_mem_usage_max_train
                        train_gpu_util[iter_index, epoch]       = gpu_util_max_train
                        
                        start_time_test = time.perf_counter()
                        test_dataloader = DataLoader( dataset = test_dataset,
                                                      batch_size = batch_size,
                                                      shuffle = False,
                                                      pin_memory = True,
                                                      num_workers = num_workers )
        
                        model.eval()  # Sets the model to evaluation mode
                        with torch.no_grad():        
                            for i, (images, labels) in enumerate(test_dataloader):
                                
                                trig_GPU_read( queue_gpu_info, event_start_read_GPU_info )
                                
                                images, labels = images.to(device), labels.to(device)  
                                
                                outputs = model(images)        
                                
                                _, predicted = torch.max(outputs.data, 1)
                                total_test += labels.size(0)
                                correct_test += (predicted == labels).sum().cpu().detach().numpy()
                                acc_test = correct_test / total_test
                                
                                data_from_GPU_queue = get_info_from_GPU_queue( queue_gpu_info, event_start_read_GPU_info)
                                if(data_from_GPU_queue != None):
                                    device_name = data_from_GPU_queue[0]
                                    device_mem_cap = data_from_GPU_queue[1]
                                    gpu_mem_usage_max_test = max(gpu_mem_usage_max_train, data_from_GPU_queue[2])
                                    gpu_util_max_test = max(gpu_util_max_train, data_from_GPU_queue[3]) 
                                
                                if( ( (i*batch_size) % 10000 ) > ( ((i+1)*batch_size) % 10000 ) or (i+1) == len(test_dataloader) ):
                                    
                                    print("Test Epoch {}/{} Batch {}/{} LR {:.6f} CorPred {}/{} Acc {:.6f} {} Mem Used {}/{} GPU Util {}"
                                            .format(    epoch,
                                                        n_epochs-1,
                                                        i,
                                                        len(test_dataloader)-1,
                                                        learning_rate,
                                                        correct_test,
                                                        total_test,
                                                        acc_test,
                                                        device_name,
                                                        gpu_mem_usage_max_test,
                                                        device_mem_cap,
                                                        gpu_util_max_test, ) )
                        
                        check_gpu_info_queue(queue_gpu_info)
                        
                        test_acc_vec[iter_index, epoch]         = acc_test
                        end_time_test                           = time.perf_counter()
                        test_time_vec[iter_index, epoch]        = end_time_test - start_time_test
                        test_gpu_mem_usage[iter_index, epoch]   = gpu_mem_usage_max_test
                        test_gpu_util[iter_index, epoch]        = gpu_util_max_test
                                
                        print("-" * 40)
                    iter_index = iter_index + 1
                    
                    clear_gpu_mem_util(model, images, labels)
    
        queue_training_results.put(  (  train_acc_vec,
                                        test_acc_vec,
                                        train_time_vec,
                                        test_time_vec,
                                        train_gpu_mem_usage,
                                        test_gpu_mem_usage,
                                        train_gpu_util,
                                        test_gpu_util,
                                        device_name,
                                        device_mem_cap, ) )


