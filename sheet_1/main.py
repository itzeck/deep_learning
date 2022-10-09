import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import LogSoftmax
import xlsxwriter

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                  ]))

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                  ]))

train_data_loader = DataLoader(
    training_data, batch_size=4
)

test_data_loader = DataLoader(
    test_data, batch_size=4
)


def train_loop(data_loader, model, loss_function, optimizer):
    n_samples = len(data_loader.dataset)
    n_batches = len(data_loader)
    loss_sum, n_correct = 0, 0
    for batch, (data, labels) in enumerate(data_loader):
        # Feed data through network and compute loss.

        prediction = model(data)
        loss = loss_function(prediction, labels)
        loss_sum += loss.item()

        n_correct += (
            (prediction.argmax(1) == labels)
            .type(torch.float)
            .sum()
            .item()
        )
        # Zero gradients.
        optimizer.zero_grad()

        # Perform backpropagation and accumulate gradients.
        loss.backward()

        # Update network parameters.
        optimizer.step()

    print(f"Train Accuracy: {n_correct / n_samples:.2%}, "f"Train Loss: {loss_sum / n_batches:.4}")
    return n_correct / n_samples, loss_sum / n_batches


def test_loop(data_loader, model, loss_function):
    n_samples = len(data_loader.dataset)
    n_batches = len(data_loader)
    loss, n_correct = 0, 0
    with torch.no_grad():
        for data, labels in data_loader:
            # Feed data through network and accumulate loss.

            prediction = model(data)
            loss += loss_function(
                prediction, labels
            ).item()

            n_correct += (
                (prediction.argmax(1) == labels)
                .type(torch.float)
                .sum()
                .item()
            )

    print(f"Test Accuracy: {n_correct / n_samples:.2%}, "f"Test Loss: {loss / n_batches:.4}")
    return n_correct / n_samples, loss / n_batches


class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(20, 64, kernel_size=(3, 3))
        self.fc1 = nn.Linear(64 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 64 * 2 * 2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.logSoftmax(x)
        return x


if __name__ == "__main__":
    """
    Tried a ResNet50 model, but it took forever
    # Initialize network, loss function, and optimizer.
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Initialize the Weight Transforms
    weights = ResNet50_Weights.DEFAULT
    preprocessing = weights.transforms()
    """

    model = mnist_model()

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate,
    )

    # Train the network.
    n_epochs = 30
    train_loss_hist = []
    train_accuracy_hist = []
    test_loss_hist = []
    test_accuracy_hist = []

    best_test_loss = 0
    tolerance = 3
    for t in range(n_epochs):
        print(f"Epoch {t + 1:02}", end=" ", flush=True)
        train_acc, train_loss = train_loop(train_data_loader, model, loss_fn, optimizer)
        test_acc, test_loss = test_loop(test_data_loader, model, loss_fn)
        train_accuracy_hist.append(train_acc)
        train_loss_hist.append(train_loss)
        test_accuracy_hist.append(test_acc)
        test_loss_hist.append(test_loss)

        # A simple form of early stopping
        if t == 0:
            best_test_loss = test_loss
        else:
            if test_loss > best_test_loss:
                tolerance -= 1
                if tolerance == 0:
                    print(f"Early stopping after three consecutive decreases in test_loss.")
                    break
            else:
                best_test_loss = test_loss
                tolerance = 3

        print(f"Best test loss: {best_test_loss:.3f}")
        print(f"Current test loss: {test_loss:.3f}")
        print(f"Tolerance: {tolerance}")

    total_list = [train_accuracy_hist, train_loss_hist, test_accuracy_hist, test_loss_hist]
    with xlsxwriter.Workbook(f"4_adam.xlsx") as workbook:
        worksheet = workbook.add_worksheet()

        for row_num, data in enumerate(total_list):
            worksheet.write_row(row_num, 0, data)

# https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
# https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#torchvision.transforms.Compose
# https://jgoepfert.pages.ub.uni-bielefeld.de/talk-deep-learning/
