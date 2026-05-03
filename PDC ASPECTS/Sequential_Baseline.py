import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from thop import profile
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import medmnist
from medmnist import PathMNIST, INFO

# ─────────────────────────────
# DEVICE
# ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Device:", DEVICE)

# ─────────────────────────────
# PATHS
# ─────────────────────────────
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
# DATASET CLASS
# Wraps MedMNIST split into a
# standard Dataset compatible
# with the rest of the pipeline
# ─────────────────────────────
SIZE = 224

tf = Compose([
    Resize((SIZE, SIZE)),
    ToTensor(),
    Normalize((0.5,)*3, (0.5,)*3)
])

class PathMNISTDataset(Dataset):
    """
    Thin wrapper around a MedMNIST PathMNIST split.
    Resizes 28×28 RGB tiles to SIZE×SIZE and returns
    a flat integer label (not a one-element array).
    """
    def __init__(self, split: str):
        # as_rgb=True guarantees 3-channel output
        self.ds = PathMNIST(split=split, download=True, as_rgb=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]          # img is a PIL Image, label is ndarray([k])
        img = tf(img)
        return img, int(label)           # scalar label, matching original pipeline


# ─────────────────────────────
# ALL TRAINING CODE INSIDE HERE
# ─────────────────────────────
if __name__ == '__main__':

    # ── PathMNIST Info ──
    info       = INFO['pathmnist']
    classes    = list(info['label'].values())   # human-readable class names
    NUM_CLASSES = len(classes)
    print(f"Classes ({NUM_CLASSES}):", classes)

    # ── Data Loading ──
    train_dataset = PathMNISTDataset(split='train')
    val_dataset   = PathMNISTDataset(split='val')
    test_dataset  = PathMNISTDataset(split='test')

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    BATCH_SIZE = 16

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=True,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=True,
        persistent_workers=False
    )

    # ── Backbones ──
    deit = timm.create_model("deit_base_patch16_224", pretrained=True)
    deit.head = nn.Identity()
    if hasattr(deit, "head_dist"):
        deit.head_dist = nn.Identity()

    medvit = MedViT_tiny(num_classes=7)
    medvit.proj_head = nn.Identity()
    ckpt_vit = torch.load(
        os.path.join(PRETRAINED, "MedViT_tiny_ISIC2018.pth"),
        map_location="cpu"
    )
    state_vit = ckpt_vit.get("model", ckpt_vit.get("state_dict", ckpt_vit))
    medvit.load_state_dict(state_vit, strict=False)
    medvit.eval()

    medformer = medformer_tiny(num_classes=4)
    medformer.head = nn.Identity()
    ckpt_former = torch.load(
        os.path.join(PRETRAINED, "Medformer_tiny_BT.pth"),
        map_location="cpu", weights_only=False
    )
    state_former = ckpt_former.get("model", ckpt_former.get("state_dict", ckpt_former))
    medformer.load_state_dict(state_former, strict=False)
    medformer.eval()

    # ── Backbone Wrappers ──
    class DeiTBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = deit
        def forward(self, x):
            return self.m.forward_features(x)[:, 0]

    class MedViTBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = medvit
        def forward(self, x):
            return self.m(x)

    class MedFormerBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = medformer
        def forward(self, x):
            x = self.m.forward_features(x)
            return x.flatten(2).mean(-1)

    # ── Fusion Model ──
    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.deit      = DeiTBackbone().to(DEVICE)
            self.medvit    = MedViTBackbone().to(DEVICE)
            self.medformer = MedFormerBackbone().to(DEVICE)

            self.p1 = nn.Linear(768, 256)
            self.p2 = nn.Linear(384, 256)
            self.p3 = nn.Linear(256, 256)

            self.cls = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, NUM_CLASSES)
            )

        def forward(self, x):
            f1 = self.p1(self.deit(x))
            f2 = self.p2(self.medvit(x))
            f3 = self.p3(self.medformer(x))
            return self.cls(torch.cat([f1, f2, f3], dim=1))

    model = Fusion().to(DEVICE)

    for m in [model.deit, model.medvit, model.medformer]:
        for p in m.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total_p:,}")

    # ── FLOPs ──
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    try:
        total_flops, _ = profile(model, inputs=(dummy,), verbose=False)
        f1_, _ = profile(model.deit,      inputs=(dummy,), verbose=False)
        f2_, _ = profile(model.medvit,    inputs=(dummy,), verbose=False)
        f3_, _ = profile(model.medformer, inputs=(dummy,), verbose=False)
        print(f"\n--- FLOPs ---")
        print(f"  DeiT        : {f1_/1e9:.2f} GFLOPs")
        print(f"  MedViT      : {f2_/1e9:.2f} GFLOPs")
        print(f"  MedFormer   : {f3_/1e9:.2f} GFLOPs")
        print(f"  Fusion+Head : {(total_flops-f1_-f2_-f3_)/1e9:.4f} GFLOPs")
        print(f"  TOTAL       : {total_flops/1e9:.2f} GFLOPs\n")
    except Exception as e:
        print(f"FLOPs error: {e}")
        total_flops = None

    # ── Latency Benchmark ──
    print("--- Latency Benchmark (Sequential) ---")
    model.eval()
    bench = torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        for _ in range(3):
            _ = model(bench)
    torch.cuda.synchronize()
    lat_times = []
    with torch.no_grad():
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _  = model(bench)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            lat_times.append((t1-t0)*1000)
    avg_latency    = np.mean(lat_times)
    avg_throughput = BATCH_SIZE / (avg_latency / 1000)
    print(f"  Latency    : {avg_latency:.2f} ms")
    print(f"  Throughput : {avg_throughput:.1f} img/s\n")

    # ── Optimizer ──
    opt = torch.optim.AdamW(
        list(model.p1.parameters()) +
        list(model.p2.parameters()) +
        list(model.p3.parameters()) +
        list(model.cls.parameters()),
        lr=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()

    # ── Evaluate Function ──
    def evaluate_full(loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        return all_labels, all_preds

    # ── Train Loop ──
    EPOCHS        = 5
    epoch_metrics = []
    total_start   = time.perf_counter()

    for e in range(EPOCHS):
        model.train()
        total_loss   = 0
        total_images = 0

        torch.cuda.synchronize()
        ep_start = time.perf_counter()

        for x, y in tqdm(train_loader, desc=f"Epoch {e+1}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss   += loss.item()
            total_images += x.size(0)

        torch.cuda.synchronize()
        ep_time    = time.perf_counter() - ep_start
        throughput = total_images / ep_time
        avg_loss   = total_loss / len(train_loader)

        val_true, val_pred = evaluate_full(val_loader)
        val_acc = accuracy_score(val_true, val_pred)
        val_f1  = f1_score(val_true, val_pred,
                           average='macro', zero_division=0)

        tflops_epoch = (
            (total_flops * len(train_loader) * 3) / 1e12
            if total_flops else float("nan")
        )

        epoch_metrics.append({
            "epoch"  : e + 1,
            "loss"   : avg_loss,
            "val_acc": val_acc,
            "val_f1" : val_f1,
            "time_s" : ep_time,
            "img_s"  : throughput,
            "tflops" : tflops_epoch,
        })

        print(
            f"Epoch {e+1}/{EPOCHS} | "
            f"Loss {avg_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val F1 {val_f1:.4f} | "
            f"Time {ep_time:.1f}s | "
            f"Throughput {throughput:.1f} img/s | "
            f"TFLOPs {tflops_epoch:.2f}"
        )

    total_time = time.perf_counter() - total_start

    # ── Final Test Evaluation ──
    y_true, y_pred = evaluate_full(test_loader)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred,    average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred,        average='macro', zero_division=0)

    # ── Final Summary ──
    print("\n" + "="*70)
    print("      SEQUENTIAL FUSION — FINAL METRICS SUMMARY")
    print("="*70)
    print(f"\n[Model]")
    if total_flops:
        print(f"  FLOPs/forward  : {total_flops/1e9:.2f} GFLOPs")
    print(f"  Latency        : {avg_latency:.2f} ms  (batch={BATCH_SIZE})")
    print(f"  Throughput     : {avg_throughput:.1f} img/s")
    print(f"  Trainable      : {trainable:,} / {total_p:,} params")
    print(f"\n[Training]")
    print(f"  Total time     : {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Avg/epoch      : {total_time/EPOCHS:.1f}s")
    print(f"\n[Per-Epoch Breakdown]")
    print(f"  {'Ep':<4} {'Loss':<9} {'ValAcc':<9} {'ValF1':<9} "
          f"{'Time(s)':<9} {'Img/s':<9} {'TFLOPs'}")
    print(f"  {'-'*62}")
    for m in epoch_metrics:
        print(
            f"  {m['epoch']:<4} "
            f"{m['loss']:<9.4f} "
            f"{m['val_acc']:<9.4f} "
            f"{m['val_f1']:<9.4f} "
            f"{m['time_s']:<9.1f} "
            f"{m['img_s']:<9.1f} "
            f"{m['tflops']:.2f}"
        )
    print(f"\n[Test Results]")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("="*70)

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Sequential Fusion (PathMNIST)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_sequential.png", dpi=150)
    plt.show()