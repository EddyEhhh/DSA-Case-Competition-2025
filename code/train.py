import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LipDataset(Dataset):
    def __init__(self, img_dir="../images", transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.heic'))]
        if any(f.lower().endswith('.heic') for f in self.image_files):
            register_heif_opener()

    def __len__(self):
        return len(self.image_files)

    def extract_hgb_from_filename(self, filename):
        match = re.search(r"_(\d+\.\d+)gdl", filename)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Could not extract HgB from filename {filename}")

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        filepath = os.path.join(self.img_dir, filename)
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.extract_hgb_from_filename(filename)
        return image, torch.tensor(label, dtype=torch.float32), filename

class HgBRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze()

def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    mean_bias = np.mean(all_preds - all_labels)
    r2 = r2_score(all_labels, all_preds)
    return mae, rmse, mean_bias, r2

def main():
    set_seed(42)
    img_dir = "./images"
    batch_size = 16
    epochs = 100
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = LipDataset(img_dir=img_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = HgBRegressor().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_mae = float('inf')

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)
        train_loss = running_train_loss / train_size
        val_mae, val_rmse, val_bias, val_r2 = evaluate_metrics(model, val_loader, device)
        scheduler.step(val_mae)
        print(f"Epoch {epoch+1}/{epochs} Train MAE: {train_loss:.4f} Val MAE: {val_mae:.4f} "
              f"RMSE: {val_rmse:.4f} Bias: {val_bias:.4f} R2: {val_r2:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "./weights/best_hgb_regressor.pt")
            print("SAVED BEST MODEL")


if __name__ == "__main__":
    main()
