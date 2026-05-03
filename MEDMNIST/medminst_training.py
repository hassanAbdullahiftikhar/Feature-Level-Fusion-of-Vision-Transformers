import torch
import torch.nn as nn
import timm
import sys
import time
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from medmnist import PathMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, ChestMNIST

# ---------------------------
# CONFIG
# ---------------------------
DATASET_NAME = "pneumoniamnist"   # 👈 ONLY CHANGE THIS

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---------------------------
# DATASET MAP
# ---------------------------
dataset_map = {
    "pathmnist": PathMNIST,
    "dermamnist": DermaMNIST,
    "octmnist": OCTMNIST,
    "pneumoniamnist": PneumoniaMNIST,
    "chestmnist": ChestMNIST,
}

num_classes_map = {
    "pathmnist": 9,
    "dermamnist": 7,
    "octmnist": 4,
    "pneumoniamnist": 2,
    "chestmnist": 14,
}

norm_stats = {
    "pathmnist"    : ([0.7405, 0.5330, 0.7058], [0.1237, 0.1751, 0.1044]),
    "dermamnist"   : ([0.7631, 0.5381, 0.5614], [0.1366, 0.1543, 0.1692]),
    "octmnist"     : ([0.1889, 0.1889, 0.1889], [0.1963, 0.1963, 0.1963]),
    "pneumoniamnist": ([0.5720, 0.5720, 0.5720], [0.1686, 0.1686, 0.1686]),
}



DatasetClass = dataset_map[DATASET_NAME]
NUM_CLASSES = num_classes_map[DATASET_NAME]

# ChestMNIST is multilabel (ignore for now unless you want BCE version)
assert DATASET_NAME != "chestmnist", "ChestMNIST needs multilabel setup"

# ---------------------------
# TRANSFORMS
# ---------------------------
mean, std = norm_stats[DATASET_NAME]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ---------------------------
# LOAD DATASET
# ---------------------------
train_ds = DatasetClass(split="train", transform=transform, download=True)
val_ds   = DatasetClass(split="val", transform=transform, download=True)
test_ds  = DatasetClass(split="test", transform=transform, download=True)

print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

# ---------------------------
# DATALOADER
# ---------------------------
def collate(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch]).squeeze().long()
    return x, y

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

# ---------------------------
# BACKBONES
# ---------------------------
MEDVIT_DIR = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\MEDVIT\MedViTV2-main"
MEDFORMER_DIR = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\MedFormer-main"

MEDVIT_CKPT = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\Pretrained Models\MedViT_tiny_ISIC2018.pth"
MEDFORMER_CKPT = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\Pretrained Models\Medformer_tiny_BT.pth"

sys.path.insert(0, MEDVIT_DIR)
sys.path.insert(0, MEDFORMER_DIR)

from MedViT import MedViT_tiny
from models.medformer import medformer_tiny

# ---------------------------
# DeiT
# ---------------------------
deit = timm.create_model("deit_base_patch16_224", pretrained=True)
deit.head = nn.Identity()

class DeiTBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        x = self.m.forward_features(x)
        return x[:, 0]

# ---------------------------
# MedViT
# ---------------------------
def load_medvit():
    model = MedViT_tiny(num_classes=7)
    ckpt = torch.load(MEDVIT_CKPT, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.proj_head = nn.Identity()

    class Backbone(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)

    return Backbone(model)

# ---------------------------
# MedFormer
# ---------------------------
def load_medformer():
    import argparse
    from torch.serialization import add_safe_globals
    add_safe_globals([argparse.Namespace])

    model = medformer_tiny(num_classes=4)
    ckpt = torch.load(MEDFORMER_CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.head = nn.Identity()

    class Backbone(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            x = self.m.forward_features(x)
            return x.flatten(2).mean(-1)

    return Backbone(model)

# ---------------------------
# FUSION MODEL (YOUR ARCHITECTURE)
# ---------------------------
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.deit = DeiTBackbone(deit).to(DEVICE)
        self.medvit = load_medvit().to(DEVICE)
        self.medformer = load_medformer().to(DEVICE)

        # freeze
        for m in [self.deit, self.medvit, self.medformer]:
            for p in m.parameters():
                p.requires_grad = False

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

        fused = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(fused)

model = FusionModel(NUM_CLASSES).to(DEVICE)

# ---------------------------
# OPTIMIZER + LOSS
# ---------------------------
optimizer = torch.optim.AdamW(
    list(model.p1.parameters()) +
    list(model.p2.parameters()) +
    list(model.p3.parameters()) +
    list(model.classifier.parameters()),
    lr=LR
)

loss_fn = nn.CrossEntropyLoss()

# ---------------------------
# TRAIN STEP
# ---------------------------
def train_epoch():
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# ---------------------------
# EVAL
# ---------------------------
def evaluate(loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)

            p = out.argmax(1).cpu().numpy()

            preds.append(p)
            labels.append(y.numpy())

    return np.concatenate(labels), np.concatenate(preds)

def metrics(loader):
    y, p = evaluate(loader)

    return (
        accuracy_score(y, p),
        precision_score(y, p, average="macro"),
        recall_score(y, p, average="macro"),
        f1_score(y, p, average="macro"),
    )

# ---------------------------
# TRAIN LOOP
# ---------------------------
start = time.time()

for epoch in range(EPOCHS):
    loss = train_epoch()
    val_acc, _, _, _ = metrics(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {loss:.4f} | Val Acc {val_acc:.4f}")

end = time.time()

# ---------------------------
# FINAL RESULTS
# ---------------------------
acc, prec, rec, f1 = metrics(test_loader)

print("\n──────── FINAL RESULTS ────────")
print("Dataset:", DATASET_NAME)
print(f"Time     : {end-start:.2f}s")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")