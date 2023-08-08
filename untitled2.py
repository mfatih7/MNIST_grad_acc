import numpy as np
import matplotlib.pyplot as plt
import os

import time

# def plot_GPU_mem_used( output_path, input_size, training_params, optimizer_types, gpu_mem_usage, device_name, device_mem_cap):
    
gpu_mem_usage_max_of_epochs = gpu_mem_usage.max(axis=1)    
gpu_mem_usage_max_of_epochs = gpu_mem_usage_max_of_epochs.reshape(-1, len(optimizer_types) )

unique_counts = [len(set(training_params[i] for training_params in training_params)) for i in range(3)]

x = np.arange(unique_counts[1], 0, -1)

for optimizer_type_id, optimizer_type in enumerate(optimizer_types):

    plt.figure()                           
    x_lines = []
    y_lines = []
    legend_names = []
    start_ind = 0
    list_ind = 0        
    
    while (True):
        x_line = []
        y_line = []
        start_ind = list_ind
        cur_exp_rat = training_params[list_ind][0]
        cur_batch = training_params[list_ind][1]
        list_ind = list_ind + 1
        while (True):
            if(cur_exp_rat == training_params[list_ind][0] and cur_batch == training_params[list_ind][1]):
                list_ind = list_ind + 1
            else:
                break
        
        # input_ress = [str(input_size*input_expand_ratio) for input_expand_ratio in input_expand_ratios]
        
        
        # for i in range( len(input_expand_ratios) ):
        #     x_lines.append( x )        
        #     y_line = []
        #     for j in range( len(batch_sizes) ):
        #         y_line.append( max( gpu_mem_usage_max_of_epochs[ list_ind : list_ind+len(optimizing_batches[j]) ] ) )
        #         list_ind = list_ind + len( optimizing_batches[j] )
        #     y_lines.append( y_line )        
        #     legend_names.append( str(input_ress[i]) + 'x' + str(input_ress[i]) )
        
        # for x_line, y_line, legend_name in zip(x_lines, y_lines, legend_names):
        #     plt.plot(x_line, y_line, label=legend_name, marker='o')
            
        # plt.grid(True)
        # plt.legend(title='Inpu Size', fontsize = 'small')
        # plt.ylabel('Average GPU Memory Usage')
        # plt.xlabel('Batch')
        # plt.title( device_name + ' ' + str(device_mem_cap) + ' MB ' + optimizer_type + ' opt' )
        
        # plt.xticks( x, [str(batch_size) for batch_size in batch_sizes] )
        
        # plot_file_name = 'gpu_mem_usage' + optimizer_type + '.png'
        # folder_path = os.path.join(output_path, device_name)
        # if not os.path.exists( folder_path ):
        #     os.makedirs(folder_path)            
        # plt.savefig( os.path.join(folder_path, plot_file_name), dpi=300)
























# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.checkpoint import checkpoint
# import torch.optim as optim

# class SubModule(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SubModule, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         return x

# class MainModel(nn.Module):
#     def __init__(self):
#         super(MainModel, self).__init__()
#         self.blocks = nn.ModuleList([SubModule(3 if i == 0 else 64, 64) for i in range(5)])
#         self.fc = nn.Linear(64 * 1 * 1, 2) # assuming input size 32x32

#     def forward(self, x):
#         for block in self.blocks:
#             x = checkpoint(block, x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)

# # Create synthetic dataset with requires_grad=True

# X_train = torch.rand(100, 3, 32, 32, requires_grad=True)
# # X_train = torch.rand(100, 3, 32, 32)

# y_train = torch.randint(0, 2, (100,))
# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# # Initialize the model
# model = MainModel()

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# epochs = 10
# for epoch in range(epochs):
#     for batch_data, batch_labels in train_loader:
#         # Forward pass
#         outputs = model(batch_data)

#         # Compute loss
#         loss = criterion(outputs, batch_labels)

#         # Zero gradients
#         optimizer.zero_grad()

#         # Backward pass
#         loss.backward()

#         # Update weights
#         optimizer.step()

#     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
