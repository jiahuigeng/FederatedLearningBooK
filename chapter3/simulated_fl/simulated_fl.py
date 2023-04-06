import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def train_client(client_data, global_model, epochs=1, lr=0.001):
    local_model = SimpleCNN()
    local_model.load_state_dict(global_model.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
    client_loader = torch.utils.data.DataLoader(client_data, batch_size=100, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        for inputs, labels in client_loader:
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return local_model.state_dict()

def evaluate_model(model, dataset):
    testloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

if __name__ == "__main__":
    trainset = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform)

    num_clients = 3
    client_datasets = []

    for i in range(num_clients):
        client_data = torch.utils.data.Subset(trainset, range(i * len(trainset) // num_clients, (i + 1) * len(trainset) // num_clients))
        client_datasets.append(client_data)


    global_model = SimpleCNN()

    num_rounds = 10
    for r in range(num_rounds):
        print(f"Round {r + 1}")
        client_weights = []

        for client_idx, client_data in enumerate(client_datasets):
            print(f"Training client {client_idx + 1}")
            client_weight = train_client(client_data, global_model)
            client_weights.append(client_weight)

        new_global_dict = {}
        for key in global_model.state_dict().keys():
            new_global_dict[key] = torch.stack([client_weights[i][key] for i in range(num_clients)], dim=0).mean(dim=0)

        global_model.load_state_dict(new_global_dict)

        # Evaluate the global model
        test_accuracy = evaluate_model(global_model, testset)
        print(f"Global model accuracy after round {r + 1}: {test_accuracy * 100:.2f}%")



