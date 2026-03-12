import os
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image

CITIES = ["Anaheim", "Bakersfield", "Los Angeles", "Riverside", "San Diego", "SLO"]

IMG_SIZE = 224


def get_model(num_classes=6):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "model_weights.pt")

    model = get_model(num_classes=len(CITIES))
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = {}
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(".jpg")]

    for filename in image_files:
        img_path = os.path.join(image_path, filename)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            results[filename] = CITIES[0]
            continue

        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred_idx = logits.argmax(dim=1).item()

        results[filename] = CITIES[pred_idx]

    return results
