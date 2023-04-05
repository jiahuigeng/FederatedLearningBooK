import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the client model (initial layers)
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        return x

# Define the server model (remaining layers)
class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='../../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

# Create client and server models
client_model = ClientModel()
server_model = ServerModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(list(client_model.parameters()) + list(server_model.parameters()), lr=0.01, momentum=0.9)

num_epochs = 10
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        # Client-side computation
        intermediate = client_model(images)

        # Send intermediate features to the server
        intermediate = intermediate.detach().requires_grad_()

        # Server-side computation
        outputs = server_model(intermediate)

        # Compute loss and update weights
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

        # Send gradients from server to client
        intermediate_grad = intermediate.grad.clone().detach()

        # Update client-side gradients
        client_model.zero_grad()
        intermediate = client_model(images)
        intermediate.backward(intermediate_grad)

        # Update weights for both client and server
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

    accuracy = correct / total * 100
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {accuracy:.2f}%')
