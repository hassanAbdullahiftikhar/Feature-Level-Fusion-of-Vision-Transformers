import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm

# ─────────────────────────────
# DEVICE
# ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Device:", DEVICE)

# ─────────────────────────────
# PATHS
# ─────────────────────────────
DATA_DIR      = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\Brain_Tumor_MRI_Classification\Brain_Tumor_MRI_Classification"
MEDVIT_DIR    = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\MEDVIT\MedViTV2-main"
MEDFORMER_DIR = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\MedFormer-main"
PRETRAINED    = r"C:\Users\hassa\OneDrive\Desktop\sem6\ANN\multimodal\Pretrained Models"

sys.path.insert(0, MEDVIT_DIR)
sys.path.insert(0, MEDFORMER_DIR)

from MedViT import MedViT_tiny
from models.medformer import medformer_tiny
import argparse
from torch.serialization import add_safe_globals
add_safe_globals([argparse.Namespace])

# ─────────────────────────────
# DATA
# ─────────────────────────────
train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")

classes     = sorted(os.listdir(train_dir))
label2id    = {c: i for i, c in enumerate(classes)}
NUM_CLASSES = len(classes)
print(f"Classes ({NUM_CLASSES}):", classes)

def load(folder):
    imgs, labels = [], []
    for c in classes:
        p = os.path.join(folder, c)
        for f in os.listdir(p):
            imgs.append(os.path.join(p, f))
            labels.append(label2id[c])
    return imgs, labels

train_imgs, train_labels = load(train_dir)
val_imgs,   val_labels   = load(val_dir)

all_imgs   = train_imgs + val_imgs
all_labels = train_labels + val_labels

x_train, x_temp, y_train, y_temp = train_test_split(
    all_imgs, all_labels,
    test_size=0.2, stratify=all_labels, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp,
    test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

# ─────────────────────────────
# DATASET
# ─────────────────────────────
SIZE = 224

tf_train = Compose([
    Resize((SIZE, SIZE)),
    ToTensor(),
    Normalize((0.5,)*3, (0.5,)*3)
])

tf_eval = Compose([
    Resize((SIZE, SIZE)),
    ToTensor(),
    Normalize((0.5,)*3, (0.5,)*3)
])

class BrainDS(Dataset):
    def __init__(self, x, y, tf):
        self.x  = x
        self.y  = y
        self.tf = tf

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = Image.open(self.x[i]).convert("RGB")
        return self.tf(img), self.y[i]

BATCH_SIZE = 16

train_loader = DataLoader(
    BrainDS(x_train, y_train, tf_train),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    BrainDS(x_val, y_val, tf_eval),
    batch_size=BATCH_SIZE,
    num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    BrainDS(x_test, y_test, tf_eval),
    batch_size=BATCH_SIZE,
    num_workers=0, pin_memory=True
)

# ─────────────────────────────
# LOAD PRETRAINED BACKBONES
# Called fresh for each config
# to avoid state bleed between runs
# ─────────────────────────────
def get_deit():
    m = timm.create_model("deit_base_patch16_224", pretrained=True)
    m.head = nn.Identity()
    if hasattr(m, "head_dist"):
        m.head_dist = nn.Identity()
    return m

def get_medvit():
    m = MedViT_tiny(num_classes=7)
    ckpt  = torch.load(
        os.path.join(PRETRAINED, "MedViT_tiny_ISIC2018.pth"),
        map_location="cpu"
    )
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    m.load_state_dict(state, strict=False)
    m.proj_head = nn.Identity()
    return m

def get_medformer():
    m = medformer_tiny(num_classes=4)
    ckpt  = torch.load(
        os.path.join(PRETRAINED, "Medformer_tiny_BT.pth"),
        map_location="cpu",
        weights_only=False
    )
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    m.load_state_dict(state, strict=False)
    m.head = nn.Identity()
    return m

# ─────────────────────────────
# BACKBONE WRAPPERS
# ─────────────────────────────
class DeiTBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m.forward_features(x)[:, 0]  # [B, 768]

class MedViTBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x)                          # [B, 384]

class MedFormerBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        x = self.m.forward_features(x)
        return x.flatten(2).mean(-1)              # [B, 256]

# ─────────────────────────────
# ABLATION MODEL DEFINITIONS
# ─────────────────────────────

# Config 1: DeiT only
class AblationDeiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DeiTBackbone(get_deit())
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.cls = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x):
        return self.cls(self.backbone(x))

# Config 2: MedViT only
class AblationMedViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MedViTBackbone(get_medvit())
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.cls = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x):
        return self.cls(self.backbone(x))

# Config 3: MedFormer only
class AblationMedFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MedFormerBackbone(get_medformer())
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )
    def forward(self, x):
        return self.cls(self.backbone(x))

# Config 4: DeiT + MedViT
class AblationDeiTMedViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.deit   = DeiTBackbone(get_deit())
        self.medvit = MedViTBackbone(get_medvit())
        for m in [self.deit, self.medvit]:
            for p in m.parameters():
                p.requires_grad = False
        self.p1  = nn.Linear(768, 256)
        self.p2  = nn.Linear(384, 256)
        self.cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x):
        f1 = self.p1(self.deit(x))
        f2 = self.p2(self.medvit(x))
        return self.cls(torch.cat([f1, f2], dim=1))

# Config 5: Full Fusion (all three)
class FullFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.deit      = DeiTBackbone(get_deit())
        self.medvit    = MedViTBackbone(get_medvit())
        self.medformer = MedFormerBackbone(get_medformer())
        for m in [self.deit, self.medvit, self.medformer]:
            for p in m.parameters():
                p.requires_grad = False
        self.p1  = nn.Linear(768, 256)
        self.p2  = nn.Linear(384, 256)
        self.p3  = nn.Linear(256, 256)
        self.cls = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
    def forward(self, x):
        f1 = self.p1(self.deit(x))
        f2 = self.p2(self.medvit(x))
        f3 = self.p3(self.medformer(x))
        return self.cls(torch.cat([f1, f2, f3], dim=1))
# Config 5: DeiT + MedFormer
class AblationDeiTMedFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.deit      = DeiTBackbone(get_deit())
        self.medformer = MedFormerBackbone(get_medformer())
        for m in [self.deit, self.medformer]:
            for p in m.parameters():
                p.requires_grad = False
        self.p1  = nn.Linear(768, 256)
        self.p3  = nn.Linear(256, 256)
        self.cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x):
        f1 = self.p1(self.deit(x))
        f3 = self.p3(self.medformer(x))
        return self.cls(torch.cat([f1, f3], dim=1))

# Config 6: MedViT2 + MedFormer
class AblationMedViTMedFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.medvit    = MedViTBackbone(get_medvit())
        self.medformer = MedFormerBackbone(get_medformer())
        for m in [self.medvit, self.medformer]:
            for p in m.parameters():
                p.requires_grad = False
        self.p2  = nn.Linear(384, 256)
        self.p3  = nn.Linear(256, 256)
        self.cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x):
        f2 = self.p2(self.medvit(x))
        f3 = self.p3(self.medformer(x))
        return self.cls(torch.cat([f2, f3], dim=1))
# ─────────────────────────────
# TRAINING AND EVAL FUNCTIONS
# ─────────────────────────────
def get_optimizer(model):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

loss_fn = nn.CrossEntropyLoss()

def train_epoch(model, optimizer):
    model.train()
    total_loss   = 0
    total_images = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss   += loss.item()
        total_images += x.size(0)

    return total_loss / len(train_loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            p = model(x).argmax(1).cpu().numpy()
            all_preds.append(p)
            all_labels.append(y.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    return {
        "acc" : accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec" : recall_score(y_true, y_pred,    average="macro", zero_division=0),
        "f1"  : f1_score(y_true, y_pred,        average="macro", zero_division=0),
        "cm"  : confusion_matrix(y_true, y_pred),
    }

# ─────────────────────────────
# RUN ONE CONFIGURATION
# ─────────────────────────────
EPOCHS = 5

def run_config(name, model):
    print(f"\n{'='*55}")
    print(f"  Running: {name}")
    print(f"{'='*55}")

    model = model.to(DEVICE)
    optimizer = get_optimizer(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / Total: {total_p:,}")

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        loss    = train_epoch(model, optimizer)
        val_m   = evaluate(model, val_loader)
        print(
            f"  Ep {epoch+1}/{EPOCHS} | "
            f"Loss {loss:.4f} | "
            f"Val Acc {val_m['acc']:.4f} | "
            f"Val F1 {val_m['f1']:.4f}"
        )

    total_time = time.perf_counter() - start
    test_m     = evaluate(model, test_loader)

    print(f"\n  --- {name} Test Results ---")
    print(f"  Accuracy  : {test_m['acc']:.4f} ({test_m['acc']*100:.2f}%)")
    print(f"  Precision : {test_m['prec']:.4f}")
    print(f"  Recall    : {test_m['rec']:.4f}")
    print(f"  F1 Score  : {test_m['f1']:.4f}")
    print(f"  Time      : {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"  Confusion Matrix:\n{test_m['cm']}")

    return {
        "name"     : name,
        "acc"      : test_m["acc"],
        "prec"     : test_m["prec"],
        "rec"      : test_m["rec"],
        "f1"       : test_m["f1"],
        "time_s"   : total_time,
        "trainable": trainable,
    }

# ─────────────────────────────
# RUN ALL CONFIGURATIONS
# ─────────────────────────────
configs = [
    ("DeiT Only",              AblationDeiT()),
    ("MedViT2 Only",           AblationMedViT()),
    ("MedFormer Only",         AblationMedFormer()),
    ("DeiT + MedViT2",         AblationDeiTMedViT()),
    ("DeiT + MedFormer",       AblationDeiTMedFormer()),
    ("MedViT2 + MedFormer",    AblationMedViTMedFormer()),
    ("Full Fusion",            FullFusion()),
]

all_results = []

for name, model in configs:
    result = run_config(name, model)
    all_results.append(result)
    # free GPU memory between configs
    del model
    torch.cuda.empty_cache()

# ─────────────────────────────
# FINAL ABLATION SUMMARY TABLE
# ─────────────────────────────
print("\n" + "="*75)
print("                    ABLATION STUDY — FINAL SUMMARY")
print("                       Dataset: Brain Tumor MRI")
print("="*75)
print(f"  {'Configuration':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Time(s)':>9} {'Trainable':>12}")
print(f"  {'-'*73}")

best_f1  = max(r["f1"]  for r in all_results)
best_acc = max(r["acc"] for r in all_results)

for r in all_results:
    acc_str = f"{r['acc']:.4f}"
    f1_str  = f"{r['f1']:.4f}"

    # mark best results
    if r["acc"] == best_acc:
        acc_str += "*"
    if r["f1"] == best_f1:
        f1_str += "*"

    print(
        f"  {r['name']:<22} "
        f"{acc_str:>8} "
        f"{r['prec']:>7.4f} "
        f"{r['rec']:>7.4f} "
        f"{f1_str:>8} "
        f"{r['time_s']:>9.1f} "
        f"{r['trainable']:>12,}"
    )

print(f"\n  * = best value in column")
print("="*75)

# ─────────────────────────────
# LATEX TABLE OUTPUT
# Copy-paste directly into paper
# ─────────────────────────────
print("\n--- LaTeX Table (copy into paper) ---\n")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Ablation Study on Brain Tumor MRI Dataset}")
print(r"\label{tab:ablation}")
print(r"\begin{tabular}{lcccc}")
print(r"\hline")
print(r"\textbf{Configuration} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} & \textbf{F1} \\")
print(r"\hline")

for r in all_results:
    bold = r["f1"] == best_f1
    name = r["name"]
    if bold:
        print(
            f"\\textbf{{{name}}} & "
            f"\\textbf{{{r['acc']:.4f}}} & "
            f"\\textbf{{{r['prec']:.4f}}} & "
            f"\\textbf{{{r['rec']:.4f}}} & "
            f"\\textbf{{{r['f1']:.4f}}} \\\\"
        )
    else:
        print(
            f"{name} & "
            f"{r['acc']:.4f} & "
            f"{r['prec']:.4f} & "
            f"{r['rec']:.4f} & "
            f"{r['f1']:.4f} \\\\"
        )

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
print("--- End LaTeX ---")