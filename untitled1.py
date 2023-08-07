import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3 if i == 0 else 64, 64, kernel_size=3, padding=1) for i in range(5)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(5)])

    def block(self, x, conv, batch_norm):
        x = conv(x)
        x = batch_norm(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        return x

    def forward(self, x):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = checkpoint(self.block, x, conv, batch_norm)
        return x.view(x.size(0), -1)

# Create synthetic dataset
X_train = torch.rand(100, 3, 32, 32, requires_grad=True)  # Added requires_grad=True
# X_train = torch.rand(100, 3, 32, 32)

y_train = torch.randint(0, 2, (100,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize the model
model = MyModel()

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
