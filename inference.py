import os
import re
import argparse
import random
import numpy as np
import pandas as pd
import torch
import time
from torchvision import transforms, models
from PIL import Image
from pillow_heif import register_heif_opener

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LipDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.heic'))]
        if any(f.lower().endswith('.heic') for f in self.image_files):
            register_heif_opener()

    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        filepath = os.path.join(self.img_dir, filename)
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, filename

class HgBRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_features, 1)
    def forward(self, x): return self.backbone(x).squeeze()

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Infer HgB from lip images")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)  # meta is not used in this code
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--weights", type=str, default="weights/best_hgb_regressor.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = LipDataset(args.images, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    model = HgBRegressor().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    all_filenames, all_preds = [], []

    start_time = time.time()
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outs = model(images)
            all_preds.extend(outs.cpu().numpy())
            all_filenames.extend(list(filenames))
    end_time = time.time()

    pd.DataFrame({"filename": all_filenames, "predicted_hgb": all_preds}).to_csv(args.out, index=False)
    print(f"Wrote predictions to {args.out}")

    total_images = len(all_filenames)
    total_time = end_time - start_time
    throughput = total_images / total_time if total_time > 0 else float('inf')
    avg_time = (total_time / total_images) * 1000 if total_images > 0 else float('inf')
    print(f"Processed {total_images} images in {total_time:.2f} seconds.")
    print(f"Throughput: {throughput:.2f} images/sec.")
    print(f"Avg inference time per image: {avg_time:.2f} ms.")

if __name__ == "__main__":
    main()
