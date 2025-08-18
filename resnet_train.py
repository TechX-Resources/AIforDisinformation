import os
import torch
from torch import nn, optim
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt

# data path
DATA_DIR = '.'  # current folder

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def rgba_loader(path):
    img = default_loader(path)
    if img.mode == 'P' or img.mode == 'LA':
        img = img.convert('RGBA')
    return img

# load dataset 
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform, loader=rgba_loader)
test_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform, loader=rgba_loader)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# load model
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last two layers: layer4 and fc
for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes (real, fake)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
train_accuracies = []

# training loop
for epoch in range(10):
    model.train()
    train_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_loader.dataset) * 100
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {acc:.2f}%")

    train_losses.append(train_loss)
    train_accuracies.append(acc)


# save model
torch.save(model, 'resnet18_full_model.pth')

# graph
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o', label='Train Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, marker='o', label='Train Accuracy', color='green')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()


