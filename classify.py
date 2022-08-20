import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import sys
import numpy as np

from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

save_path = './model/model.ckpt'
if not os.path.exists('model'):
    os.mkdir('model')


class CIFARClassifier(nn.Module):
    def __init__(self, in_dim, hidden1, hidden2, out_dim):
        super(CIFARClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1), nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2), nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def train():
    data_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_ds = datasets.CIFAR10(root='./datasets', train=True, transform=data_tf, download=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True)
    test_ds = datasets.CIFAR10(root='./datasets', train=False, transform=data_tf, download=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=True
    )
    model = CIFARClassifier(3 * 32 * 32, 300, 100, 10)
    model.to(device)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("Loop{: ^15}{: ^15}{: ^15}{: ^15}".
          format('Train Loss',
                 'Train Acc %',
                 'Test Loss',
                 'Test Acc %'))

    for epoch in range(10):
        model.train()
        train_losses = 0.0
        train_counter = 0
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses += loss.item() * data.size(0)
            _, pred = out.max(1)
            train_correct += (pred == target).sum().item()
            train_counter += data.shape[0]
        test_losses = 0.0
        test_counter = 0
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            target = target.to(device)
            data = data.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_losses += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            test_correct += (pred == target).sum().item()
            test_counter += data.size(0)
        print('{}/10{: ^15}{: ^15}{: ^15}{: ^15}'.
              format(epoch + 1,
                     round(train_losses / train_counter, ndigits=4),
                     round(train_correct / train_counter * 100, ndigits=4),
                     round(test_losses / test_counter, ndigits=4),
                     round(test_correct / test_counter * 100, ndigits=4)))
    torch.save(model.state_dict(), save_path)
    print('Model saved in file: {}'.format(save_path))


def test(img_file: str):
    model = CIFARClassifier(3 * 32 * 32, 300, 100, 10)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    else:
        print('No trained model!')
        exit(1)
    model.to(device)
    model.eval()
    data_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(img_file)
    image = image.resize((32, 32))
    tensor = data_tf(image)
    tensor = tensor.reshape((1, 3 * 32 * 32))

    tensor = tensor.to(device)
    output = model(tensor)
    _, pred = torch.max(output.data, 1)
    print('prediction result: {}'.format(labels[pred]))


def main():
    if sys.argv[1] == 'train':
        train()
    else:
        test(sys.argv[2])


if __name__ == '__main__':
    main()
