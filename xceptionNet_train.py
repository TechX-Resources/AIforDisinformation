import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
DATA_DIR = "data/train"
TEST_DIR = "data/test"
NUM_CLASSES = 2
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda")

model = create_model("xception", pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = nn.DataParallel(model)
model = model.to(DEVICE)

print("Preparing model...")
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

print("Preparing data loaders...")
train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

print(f"Total training samples: {len(train_dataset)}")
print(f"Total test samples: {len(test_dataset)}")
print(f"Class mapping: {train_dataset.class_to_idx}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses = []
test_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
    test_acc = correct_test / total_test * 100
    train_losses.append(total_loss)
    test_accuracies.append(test_acc)
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

torch.save(model.state_dict(), "xception_deepfake.pth")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker="o")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, marker="o", color="green")
plt.title("Test Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
