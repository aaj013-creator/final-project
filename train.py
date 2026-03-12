import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from PIL import Image

IMG_SIZE = 224
CITIES = ["Anaheim", "Bakersfield", "Los Angeles", "Riverside", "San Diego", "SLO"]
CITY_TO_IDX = {c: i for i, c in enumerate(CITIES)}


def parse_label(filename):
    name = os.path.splitext(filename)[0]
    parts = name.rsplit("-", 1)
    if len(parts) == 2:
        return parts[0]
    return None


class SoCalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.samples = []
        for f in os.listdir(root_dir):
            if f.lower().endswith(".jpg"):
                city = parse_label(f)
                if city and city in CITY_TO_IDX:
                    self.samples.append((f, CITY_TO_IDX[city]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = os.path.join(self.root, filename)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="model_weights.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SoCalDataset(args.data, transform=transform_train)
    if len(dataset) == 0:
        raise SystemExit("No images found. Extract data.zip into a folder and pass --data <path>.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CITIES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        n = len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs}  Loss: {total_loss / n:.4f}")

    save_path = args.out
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")
    print("Zip predict.py and model_weights.pt for Gradescope (top-level, < 50MB).")


if __name__ == "__main__":
    main()
