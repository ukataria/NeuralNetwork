import torch
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
num_epochs = 5
batch_size = 1
learning_rate = 0.001
D_in, H1, H2, D_out = 784, 64, 32, 10


def convertToOneHot(vector, num_classes=None):
    result = np.zeros((len(vector), num_classes), dtype='int32')
    result[np.arange(len(vector)), vector] = 1
    return result


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='mnist_data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='mnist_data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, D_out, bias=True),
    torch.nn.Sigmoid(),
)

# Loss function
criterion = torch.nn.MSELoss(reduction='sum')
accuracy_train = torch.zeros(num_epochs)
accuracy_test = torch.zeros(num_epochs)
# Train the model
total_step = len(train_loader)
first = time.time()
for epoch in range(num_epochs):
    correct_train = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate Loss and Accuracy
        t = convertToOneHot(labels, 10)
        labels = torch.tensor(t, dtype=torch.float32)
        loss = criterion(outputs, labels)
        pred_train = torch.argmax(outputs, axis=1)
        true_train = torch.argmax(labels, axis=1)
        for po in range(len(pred_train)):
            if pred_train[po] == true_train[po]:
                correct_train = correct_train + 1

        # Backprpagation
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    # Training Accuracy
    accuracy_train[epoch] = correct_train / 60000
    print("Training accuracy:")
    print(accuracy_train[epoch])

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            out = model(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy_test[epoch] = correct/total
        print("Test accuracy:")
        print(accuracy_test[epoch])

print("execution time:", time.time() - first)