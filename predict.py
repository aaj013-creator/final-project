import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

def build_model(num_classes):
    model = models.resnet18(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = os.path.join(os.path.dirname(__file__), "socal_model.pt")
    checkpoint = torch.load(model_file, map_location=device)

    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    predictions = {}

    image_files = [
        f for f in os.listdir(image_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()

    with torch.no_grad():
        for filename in image_files:
            full_path = os.path.join(image_path, filename)

            image = Image.open(full_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)

            logits = model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_city = class_names[pred_idx]

            predictions[filename] = pred_city

    return predictions
