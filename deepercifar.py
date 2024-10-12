import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

# Depthwise convolution with binarized filters
class MaxBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        max_val, _ = weight.view(weight.size(0), -1).max(dim=1)
        binarized_weight = (weight == max_val.view(-1, 1, 1, 1)).float()
        return binarized_weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# Depthwise Morphological Layer with depthwise pooling
class DepthwiseMorphological(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DepthwiseMorphological, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2)

    def forward(self, x):
        binarized_weight = MaxBinarize.apply(self.depthwise_conv.weight)
        x = F.conv2d(x, binarized_weight, groups=x.size(1), padding=self.depthwise_conv.padding)
        return x

# Erosion Layer (min-pooling after depthwise convolution)
class ErosionLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ErosionLayer, self).__init__()
        self.erosion = DepthwiseMorphological(in_channels, kernel_size)

    def forward(self, x):
        return F.max_pool2d(self.erosion(x), kernel_size=3, stride=1, padding=1)

# Dilation Layer (max-pooling after depthwise convolution)
class DilationLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DilationLayer, self).__init__()
        self.dilation = DepthwiseMorphological(in_channels, kernel_size)

    def forward(self, x):
        return F.max_pool2d(-self.dilation(-x), kernel_size=3, stride=1, padding=1)

# Combined morphological operations (erosion followed by dilation)
class MorphologicalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MorphologicalLayer, self).__init__()
        self.erosion = ErosionLayer(in_channels, kernel_size)
        self.dilation = DilationLayer(in_channels, kernel_size)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)  # Batch normalization

    def forward(self, x):
        x_erosion = self.erosion(x)
        x_dilation = self.dilation(x)
        x_combined = self.pointwise_conv(x_erosion + x_dilation)
        x_combined = self.batch_norm(x_combined)
        return F.relu(x_combined)  # Apply ReLU for non-linearity

# Deeper DeepMorphNet architecture (slightly reduced depth and neurons)
class DeepMorphNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepMorphNet, self).__init__()

        # Reduced depth and slightly fewer neurons
        self.layer1 = MorphologicalLayer(3, 64, kernel_size=5)
        self.layer2 = MorphologicalLayer(64, 128, kernel_size=5)
        self.layer3 = MorphologicalLayer(128, 256, kernel_size=3)

        # Fully connected layers with fewer neurons
        self.flatten_size = None  # To be set dynamically
        self.fc1 = None  # Will be initialized later
        self.fc2 = nn.Linear(1024, 512)  # Reduced neurons
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Dynamically calculate the size of the flattened input
        if self.flatten_size is None:
            self.flatten_size = x.view(x.size(0), -1).shape[1]
            self.fc1 = nn.Linear(self.flatten_size, 1024).to(x.device)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout before the final layers
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training and validation functions
def train(model, trainloader, valloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_accuracy = test(model, valloader, device)
    return running_loss / len(trainloader), val_accuracy

def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Main execution
if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001  # Lower learning rate

    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Split the training set into train and validation sets (e.g., 80/20)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepMorphNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Switched to Adam

    # Training loop
    for epoch in range(num_epochs):
        train_loss, val_accuracy = train(model, trainloader, valloader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Final test accuracy after training
    test_accuracy = test(model, testloader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    print('Training finished.')
