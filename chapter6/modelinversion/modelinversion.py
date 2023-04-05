import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the target model (a simple CNN)
class TargetModel(nn.Module):
    def __init__(self):
        super(TargetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Define the attacker model (another simple CNN)
class AttackerModel(nn.Module):
    def __init__(self):
        super(AttackerModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 784)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x.view(-1, 1, 28, 28)

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Instantiate and train the target model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_model = TargetModel().to(device)
attacker_model = AttackerModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(target_model.parameters(), lr=0.001)

# Train the target model
target_model.train()
for epoch in range(5):
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = target_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

attacker_criterion = nn.MSELoss()
attacker_optimizer = optim.Adam(attacker_model.parameters(), lr=0.001)

target_model.eval()
attacker_model.train()
for epoch in range(5):
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            target_outputs = target_model(images)

        attacker_optimizer.zero_grad()
        attacker_images = attacker_model(target_outputs)
        attacker_loss = attacker_criterion(attacker_images, images)
        attacker_loss.backward()
        attacker_optimizer.step()

# Test the attacker model
attacker_model.eval()
total_loss = 0.0
num_samples = 0

for i, (images, labels) in enumerate(testloader):
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        target_outputs = target_model(images)
        attacker_images = attacker_model(target_outputs)
        loss = attacker_criterion(attacker_images, images)
        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

average_loss = total_loss / num_samples
print("Average loss of the attacker model: {:.4f}".format(average_loss))

import matplotlib.pyplot as plt

def imshow(img, title=None):
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = (0.3081 * img) + 0.1307  # Un-normalize the image
    img = img.clip(0, 1)  # Clip the values to the valid range for imshow
    plt.imshow(img.squeeze(), cmap='gray')
    if title is not None:
        plt.title(title)

# Choose a sample from the test dataset
sample_idx = 1
image, label = testset[sample_idx]
image = image.to(device)
image = image.unsqueeze(0)

# Get the target model's prediction and the attacker model's reconstructed image
with torch.no_grad():
    target_output = target_model(image)
    attacker_image = attacker_model(target_output)
    pred_label = torch.argmax(target_output, dim=1)

# Visualize the results
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 3, 1)
imshow(image[0], title="Original (Label: {})".format(label))
ax2 = fig.add_subplot(1, 3, 2)
imshow(attacker_image[0], title="Reconstructed (Predicted: {})".format(pred_label.item()))

plt.show()