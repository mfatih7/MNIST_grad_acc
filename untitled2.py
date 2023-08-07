import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
import torch.optim as optim

class SubModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SubModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.blocks = nn.ModuleList([SubModule(3 if i == 0 else 64, 64) for i in range(5)])
        self.fc = nn.Linear(64 * 1 * 1, 2) # assuming input size 32x32

    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(block, x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Create synthetic dataset with requires_grad=True

X_train = torch.rand(100, 3, 32, 32, requires_grad=True)
# X_train = torch.rand(100, 3, 32, 32)

y_train = torch.randint(0, 2, (100,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize the model
model = MainModel()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch_data, batch_labels in train_loader:
        # Forward pass
        outputs = model(batch_data)

        # Compute loss
        loss = criterion(outputs, batch_labels)

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
