import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

data_path = 'D:/data'

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
                               batch_size = 64,
                               shuffle = True )

test_dataloader = DataLoader( dataset = test_dataset,
                               batch_size = 64,
                               shuffle = True )

train_features, train_labels = next(iter(train_dataloader))
train_features_num = train_features.detach().numpy()

test_features, test_labels = next(iter(test_dataloader))
test_features_num = test_features.detach().numpy()


# x = train_dataset.__getitem__(0)[0]
# y = train_dataset.__getitem__(0)[1]

# x_num = x.detach().numpy()
# y_num = y.detach().numpy()



# a = train_dataset[100][0][:]
# b = train_dataset[10][:][:]
# plt.imshow(train_dataset[0][0][0])

# print(a.size())
# print(len(b))

# x_len = train_dataset.__len__()

# A = train_dataset.data
# A_lab = train_dataset.targets

# batch_size = 10
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()