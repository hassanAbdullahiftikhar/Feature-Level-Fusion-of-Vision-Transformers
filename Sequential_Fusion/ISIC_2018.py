import os
import time
import sys
import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from thop import profile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision.transforms import (
    Compose, Resize, CenterCrop,
    RandomHorizontalFlip, RandomRotation,
    ToTensor, Normalize
)
from torch.utils.data import DataLoader
from PIL import Image

# ─────────────────────────────
# DEVICE
# ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ─────────────────────────────
# PATHS
# ─────────────────────────────
ROOT = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\ISIC2018"

TRAIN_DIR = os.path.join(ROOT, "ISIC2018_Train", "Categorized")
TEST_DIR  = os.path.join(ROOT, "ISIC2018_Test", "Categorized")

MEDVIT_DIR = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\MEDVIT\MedViTV2-main"
MEDFORMER_DIR = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\MedFormer-main"

sys.path.insert(0, MEDVIT_DIR)
sys.path.insert(0, MEDFORMER_DIR)

from MedViT import MedViT_tiny
from models.medformer import medformer_tiny

# ─────────────────────────────
# LOAD DATA
# ─────────────────────────────
classes = sorted([c for c in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, c))])
label2id = {c: i for i, c in enumerate(classes)}

def load_folder(folder):
    imgs, labels = [], []
    for c in classes:
        path = os.path.join(folder, c)
        for f in os.listdir(path):
            imgs.append(os.path.join(path, f))
            labels.append(label2id[c])
    return imgs, labels

train_imgs, train_labels = load_folder(TRAIN_DIR)
test_imgs, test_labels   = load_folder(TEST_DIR)

all_imgs = train_imgs + test_imgs
all_labels = train_labels + test_labels

# Stratified split (70-15-15)
x_train, x_temp, y_train, y_temp = train_test_split(
    all_imgs, all_labels, test_size=0.2,
    stratify=all_labels, random_state=42
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5,
    stratify=y_temp, random_state=42
)

print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

# ─────────────────────────────
# TRANSFORMS
# ─────────────────────────────
SIZE = 224

train_tf = Compose([
    Resize((SIZE, SIZE)),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ToTensor(),
    Normalize(mean=(0.5,)*3, std=(0.5,)*3),
])

eval_tf = Compose([
    Resize((SIZE, SIZE)),
    CenterCrop(SIZE),
    ToTensor(),
    Normalize(mean=(0.5,)*3, std=(0.5,)*3),
])

# ─────────────────────────────
# DATASET
# ─────────────────────────────
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, tf):
        self.x = x
        self.y = y
        self.tf = tf

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = Image.open(self.x[idx]).convert("RGB")
        return self.tf(img), self.y[idx]

train_loader = DataLoader(CustomDataset(x_train, y_train, train_tf), batch_size=16, shuffle=True)
val_loader   = DataLoader(CustomDataset(x_val, y_val, eval_tf), batch_size=16)
test_loader  = DataLoader(CustomDataset(x_test, y_test, eval_tf), batch_size=16)

# ─────────────────────────────
# MODELS
# ─────────────────────────────
deit = timm.create_model("deit_base_patch16_224", pretrained=True)
deit.head = nn.Identity()

class DeiTBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m.forward_features(x)[:, 0]

# MedViT
medvit = MedViT_tiny(num_classes=7)
medvit.proj_head = nn.Identity()
medvit.load_state_dict(torch.load(
    r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\Pretrained Models\MedViT_tiny_ISIC2018.pth",
    map_location=DEVICE
), strict=False)

class MedViTBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x)

# MedFormer
medformer = medformer_tiny(num_classes=4)
medformer.head = nn.Identity()
medformer.load_state_dict(torch.load(
    r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\Pretrained Models\Medformer_tiny_BT.pth",
    map_location=DEVICE,
    weights_only=False
), strict=False)

class MedFormerBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        x = self.m.forward_features(x)
        return x.flatten(2).mean(-1)

# ─────────────────────────────
# FUSION MODEL
# ─────────────────────────────
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.deit = DeiTBackbone(deit).to(DEVICE)
        self.medvit = MedViTBackbone(medvit).to(DEVICE)
        self.medformer = MedFormerBackbone(medformer).to(DEVICE)

        self.p1 = nn.Linear(768, 256)
        self.p2 = nn.Linear(384, 256)
        self.p3 = nn.Linear(256, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.p1(self.deit(x))
        f2 = self.p2(self.medvit(x))
        f3 = self.p3(self.medformer(x))
        return self.classifier(torch.cat([f1, f2, f3], dim=1))

model = FusionModel(len(classes)).to(DEVICE)

# Freeze backbones
for m in [model.deit, model.medvit, model.medformer]:
    for p in m.parameters():
        p.requires_grad = False

# ─────────────────────────────
# TRAINING SETUP
# ─────────────────────────────
opt = torch.optim.AdamW(
    list(model.p1.parameters()) +
    list(model.p2.parameters()) +
    list(model.p3.parameters()) +
    list(model.classifier.parameters()),
    lr=1e-4
)

criterion = nn.CrossEntropyLoss()

# ─────────────────────────────
# TRAIN FUNCTION
# ─────────────────────────────
def train_epoch():
    model.train()
    total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

        total += loss.item()

    return total / len(train_loader)

# ─────────────────────────────
# EVALUATION
# ─────────────────────────────
def evaluate(loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

def evaluate_full(loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    return y_true, y_pred

# ─────────────────────────────
# TRAIN LOOP
# ─────────────────────────────
start = time.time()

for epoch in range(5):
    loss = train_epoch()
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1} | Loss {loss:.4f} | Val Acc {val_acc:.4f}")

end = time.time()

# ─────────────────────────────
# FINAL METRICS
# ─────────────────────────────
y_true, y_pred = evaluate_full(test_loader)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted")
rec  = recall_score(y_true, y_pred, average="weighted")
f1   = f1_score(y_true, y_pred, average="weighted")

print("\n──────── FINAL RESULTS ────────")
print("Training Time:", end - start)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# ─────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()