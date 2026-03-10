import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
MODEL_PATH = "socal_model.pt"
CURVE_PATH = "training_curve.png"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SoCalDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_files.sort()
        self.labels = [self._extract_label(fname) for fname in self.image_files]
        self.class_names = sorted(list(set(self.labels)))
        self.class_to_idx = {city: idx for idx, city in enumerate(self.class_names)}

    def _extract_label(self, filename):
        return filename.split("-")[0].lower()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx

def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_examples += X.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_examples += X.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc

def plot_training_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.plot(range(1, len(val_losses) + 1), val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Empirical Risk / Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    set_seed(RANDOM_STATE)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_dataset = SoCalDataset(DATA_DIR, transform=None)

    indices = np.arange(len(full_dataset))
    labels = np.array([full_dataset.class_to_idx[label] for label in full_dataset.labels])

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    train_dataset = SoCalDataset(DATA_DIR, transform=train_transform)
    val_dataset = SoCalDataset(DATA_DIR, transform=val_transform)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.class_names
    model = build_model(num_classes=len(class_names)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    train_losses = []
    val_losses = []
    best_val_acc = -1.0

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "img_size": IMG_SIZE
                },
                MODEL_PATH
            )

    elapsed = time.time() - start_time
    plot_training_curve(train_losses, val_losses, CURVE_PATH)

    print("Training finished.")
    print(best_val_acc)
    print(elapsed / 60)

if __name__ == "__main__":
    main()
