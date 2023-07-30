import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from models.models import get_model

data_path = './'

# Hyperparameters
batch_size = 64
learning_rate = 0.001
n_epochs = 5
optimizer_type = 'Adam'
# optimizer_type = 'SGD'


transform = transforms.Compose([
    transforms.ToTensor(),           # Convert PIL images to tensors
    transforms.Normalize((0.5,), (0.5,))   # Normalize the image data to the range [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(     root = data_path,
                                               train = True,
                                               download = True, 
                                               transform = transform )


test_dataset = torchvision.datasets.MNIST(     root = data_path,
                                               train = False,
                                               download = True, 
                                               transform = transform )

train_dataloader = DataLoader( dataset = train_dataset,
                               batch_size = batch_size,
                               shuffle = True )

test_dataloader = DataLoader( dataset = test_dataset,
                               batch_size = batch_size,
                               shuffle = True )

train_features, train_labels = next(iter(train_dataloader))
train_features_num = train_features.detach().numpy()

test_features, test_labels = next(iter(test_dataloader))
test_features_num = test_features.detach().numpy()
        
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model().to(device)
criterion = nn.CrossEntropyLoss()

if(optimizer_type == 'Adam'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop

train_acc_vec = np.zeros( n_epochs )
test_acc_vec = np.zeros( n_epochs )

for epoch in range(n_epochs):
    
    correct_train = 0
    total_train = 0
    
    correct_test = 0
    total_test = 0
    
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

        if (i + 1) % 100 == 0:
        
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
    train_acc_vec[epoch] = acc_train

    model.eval()  # Sets the model to evaluation mode
    with torch.no_grad():        
        for i, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(device), labels.to(device)       
            
            outputs = model(images)        
            
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().cpu().detach().numpy()
            acc_test = correct_test / total_test
    
            if (i + 1) % 100 == 0:
            
                print("Test Epoch {}/{} Batch {}/{} LR {:.6f} CorPred {}/{} Acc {:.6f}"
                        .format(    epoch,
                                    n_epochs-1,
                                    i,
                                    len(test_dataloader)-1,
                                    learning_rate,
                                    correct_test,
                                    total_test,
                                    acc_test) )
        test_acc_vec[epoch] = acc_test
            
    print("-" * 80)