import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir1, path_dir2):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = self.filter_valid_images(sorted(os.listdir(path_dir1)), path_dir1)
        self.dir2_list = self.filter_valid_images(sorted(os.listdir(path_dir2)), path_dir2)

    def filter_valid_images(self, img_list, path_dir):
        valid_images = []
        for img_name in img_list:
            img_path = os.path.join(path_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None and img.size != 0:
                valid_images.append(img_name)
        return valid_images

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)

    def __getitem__(self, idx):
        idx = idx % self.__len__()

        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        while img is None or img.size == 0:
            idx = (idx + 1) % self.__len__()
            if idx < len(self.dir1_list):
                class_id = 0
                img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
            else:
                class_id = 1
                idx -= len(self.dir1_list)
                img_path = os.path.join(self.path_dir2, self.dir2_list[idx])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if idx == 0:
                raise IndexError("All images are empty.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0

        img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
        img = img.transpose((2, 0, 1))
        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)
        return {'img': t_img, 'label': t_class_id}

train_cats_path = "/content/gdrive/My Drive/Colab Notebooks/dataset/train/Cats/"
train_dogs_path = "/content/gdrive/My Drive/Colab Notebooks/dataset/train/Dogs/"
test_cats_path = "/content/gdrive/My Drive/Colab Notebooks/dataset/test/Cats/"
test_dogs_path = "/content/gdrive/My Drive/Colab Notebooks/dataset/test/Dogs/"

train_ds_catsdogs = Dataset2class(train_dogs_path, train_cats_path)
test_ds_catsdogs = Dataset2class(test_dogs_path, test_cats_path)

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs, shuffle=True, drop_last=True,
    batch_size=batch_size, num_workers=0)

test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs, shuffle=True, drop_last=False,
    batch_size=batch_size, num_workers=0)

model = tv.models.resnet34()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(train_loader):
        inputs, labels = batch['img'].to(device), batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, labels = batch['img'].to(device), batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# training loop
num_epochs = 10
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")


torch.save(model.state_dict(), "cats_dogs_resnet34.pth")

# testing loop
test_losses = []
test_accuracies = []

model.load_state_dict(torch.load("cats_dogs_resnet34.pth"))

test_loss, test_acc = test(model, test_loader, criterion, device)
test_losses.append(test_loss)
test_accuracies.append(test_acc)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")





