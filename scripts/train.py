# scripts/train.py
import os
import random
import shutil
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# === CONFIG ===
DATA_DIR = r"D:\New folder\sign\ASL-Fingerspelling\dataset"  # Your dataset root (A-Z folders here)
SPLIT_RATIO = 0.8  # Train/Test split ratio
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
USE_CUDA = torch.cuda.is_available()

train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

# === 1. AUTO-SPLIT DATASET ===
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    print("Splitting dataset into train/test...")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_path) or class_name in ["train", "test"]:
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in test_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

    print("Dataset split completed.")
else:
    print("Train/Test folders already exist. Skipping split.")

# === 2. TRANSFORMS (with advanced augmentation for better accuracy) ===
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === 3. LOAD DATA ===
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === 4. MODEL (Transfer Learning for better accuracy) ===
device = torch.device("cuda" if USE_CUDA else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Using ResNet50 for better performance
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === 5. TRAIN LOOP ===
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}")

# === 6. EVALUATION ===
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# === 7. SAVE MODEL ===
torch.save(model.state_dict(), "asl_model.pth")
print("Model saved as asl_model.pth")