import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from thop import profile
from tqdm import tqdm
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

# ─────────────────────────────
# PathMNIST INFO
# ─────────────────────────────
info        = INFO['pathmnist']
classes     = list(info['label'].values())   # 9 colon-histology class names
NUM_CLASSES = len(classes)
print(f"Classes ({NUM_CLASSES}):", classes)

# ─────────────────────────────
# DATASET & LOADERS
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

class PathMNISTDataset(Dataset):
    """
    Thin wrapper around a MedMNIST PathMNIST split.
    Resizes 28×28 RGB tiles to SIZE×SIZE and returns
    a flat integer label (not a one-element array).
    """
    def __init__(self, split: str, transform):
        self.ds = PathMNIST(split=split, download=True, as_rgb=True)
        self.tf = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]     # img: PIL Image, label: ndarray([k])
        return self.tf(img), int(label)

BATCH_SIZE = 16

train_loader = DataLoader(
    PathMNISTDataset(split='train', transform=tf_train),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True,
    persistent_workers=False
)
val_loader = DataLoader(
    PathMNISTDataset(split='val', transform=tf_eval),
    batch_size=BATCH_SIZE,
    num_workers=0, pin_memory=True,
    persistent_workers=False
)
test_loader = DataLoader(
    PathMNISTDataset(split='test', transform=tf_eval),
    batch_size=BATCH_SIZE,
    num_workers=0, pin_memory=True,
    persistent_workers=False
)

print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

# ─────────────────────────────
# BACKBONES
# ─────────────────────────────

# ── DeiT ──
deit = timm.create_model("deit_base_patch16_224", pretrained=True)
deit.head = nn.Identity()
if hasattr(deit, "head_dist"):
    deit.head_dist = nn.Identity()

# ── MedViT ──
medvit = MedViT_tiny(num_classes=7)
ckpt_vit = torch.load(
    os.path.join(PRETRAINED, "MedViT_tiny_ISIC2018.pth"),
    map_location="cpu"
)
state_vit = ckpt_vit.get("model", ckpt_vit.get("state_dict", ckpt_vit))
medvit.load_state_dict(state_vit, strict=False)
medvit.proj_head = nn.Identity()
medvit.eval()
print("MedViT weights loaded")

# ── MedFormer ──
import argparse
from torch.serialization import add_safe_globals
add_safe_globals([argparse.Namespace])

medformer = medformer_tiny(num_classes=4)
ckpt_former = torch.load(
    os.path.join(PRETRAINED, "Medformer_tiny_BT.pth"),
    map_location="cpu",
    weights_only=False
)
state_former = ckpt_former.get("model", ckpt_former.get("state_dict", ckpt_former))
medformer.load_state_dict(state_former, strict=False)
medformer.head = nn.Identity()
medformer.eval()
print("MedFormer weights loaded")

# ─────────────────────────────
# BACKBONE WRAPPERS
# ─────────────────────────────
class DeiTBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m.forward_features(x)[:, 0]  # CLS token [B, 768]

class MedViTBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x)

class MedFormerBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        x = self.m.forward_features(x)
        return x.flatten(2).mean(-1)

# ─────────────────────────────
# VERIFY DIMS
# ─────────────────────────────
print("\n--- Verifying backbone output dimensions ---")
_dummy = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    deit_dim      = DeiTBackbone(deit)(_dummy).shape[-1]
    medvit_dim    = MedViTBackbone(medvit)(_dummy).shape[-1]
    medformer_dim = MedFormerBackbone(medformer)(_dummy).shape[-1]

print(f"  DeiT      → {deit_dim}")
print(f"  MedViT    → {medvit_dim}")
print(f"  MedFormer → {medformer_dim}")
print("--------------------------------------------\n")

COMMON_DIM = 256
FUSED_DIM  = COMMON_DIM * 3

# ─────────────────────────────
# STREAM FUSION MODEL
# ─────────────────────────────
class StreamFusionModel(nn.Module):
    def __init__(self, num_classes, deit_dim, medvit_dim, medformer_dim):
        super().__init__()

        # Backbones
        self.deit      = DeiTBackbone(deit)
        self.medvit    = MedViTBackbone(medvit)
        self.medformer = MedFormerBackbone(medformer)

        # Freeze all backbone parameters
        for backbone in [self.deit, self.medvit, self.medformer]:
            for p in backbone.parameters():
                p.requires_grad = False

        # Verified projection layers
        self.p1 = nn.Linear(deit_dim,      COMMON_DIM)
        self.p2 = nn.Linear(medvit_dim,    COMMON_DIM)
        self.p3 = nn.Linear(medformer_dim, COMMON_DIM)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(FUSED_DIM),
            nn.Linear(FUSED_DIM, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # One dedicated stream per backbone
        self.s1 = torch.cuda.Stream()
        self.s2 = torch.cuda.Stream()
        self.s3 = torch.cuda.Stream()

    def forward(self, x):
        # Force backbones to eval regardless of model.train()
        # Required: frozen params + BatchNorm stability
        self.deit.m.eval()
        self.medvit.m.eval()
        self.medformer.m.eval()

        f1 = f2 = f3 = None

        # All three backbone forward passes launched concurrently
        # No microbatch loop — full batch through all streams at once
        with torch.cuda.stream(self.s1):
            f1 = self.p1(self.deit(x))

        with torch.cuda.stream(self.s2):
            f2 = self.p2(self.medvit(x))

        with torch.cuda.stream(self.s3):
            f3 = self.p3(self.medformer(x))

        # Wait for all three streams before fusion
        torch.cuda.current_stream().wait_stream(self.s1)
        torch.cuda.current_stream().wait_stream(self.s2)
        torch.cuda.current_stream().wait_stream(self.s3)

        fused = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(fused)

model = StreamFusionModel(
    num_classes=NUM_CLASSES,
    deit_dim=deit_dim,
    medvit_dim=medvit_dim,
    medformer_dim=medformer_dim
).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / Total: {total_p:,}\n")

# ─────────────────────────────
# FLOPs
# ─────────────────────────────
print("--- FLOPs Analysis ---")
dummy_gpu = torch.randn(1, 3, 224, 224).to(DEVICE)

try:
    total_flops, _ = profile(model, inputs=(dummy_gpu,), verbose=False)
    f1_, _ = profile(model.deit,      inputs=(dummy_gpu,), verbose=False)
    f2_, _ = profile(model.medvit,    inputs=(dummy_gpu,), verbose=False)
    f3_, _ = profile(model.medformer, inputs=(dummy_gpu,), verbose=False)

    print(f"  DeiT        : {f1_/1e9:.2f} GFLOPs")
    print(f"  MedViT      : {f2_/1e9:.2f} GFLOPs")
    print(f"  MedFormer   : {f3_/1e9:.2f} GFLOPs")
    print(f"  Fusion+Head : {(total_flops-f1_-f2_-f3_)/1e9:.4f} GFLOPs")
    print(f"  TOTAL       : {total_flops/1e9:.2f} GFLOPs")
except Exception as e:
    print(f"FLOPs error: {e}")
    total_flops = None
print("----------------------\n")

# ─────────────────────────────
# LATENCY BENCHMARK
# ─────────────────────────────
print("--- Latency Benchmark (Stream Model) ---")
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
        lat_times.append((t1 - t0) * 1000)

avg_latency    = np.mean(lat_times)
avg_throughput = BATCH_SIZE / (avg_latency / 1000)
print(f"  Avg latency  : {avg_latency:.2f} ms  (batch={BATCH_SIZE})")
print(f"  Throughput   : {avg_throughput:.1f} images/sec")
print("----------------------------------------\n")

# ─────────────────────────────
# OPTIMIZER — unfrozen only
# ─────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
loss_fn = nn.CrossEntropyLoss()

# ─────────────────────────────
# EVALUATE FUNCTION
# ─────────────────────────────
def evaluate(loader):
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
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall"   : recall_score(y_true, y_pred,    average="macro", zero_division=0),
        "f1"       : f1_score(y_true, y_pred,        average="macro", zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred),
    }

# ─────────────────────────────
# TRAIN LOOP WITH FULL METRICS
# ─────────────────────────────
EPOCHS        = 5
epoch_metrics = []
total_start   = time.perf_counter()

for epoch in range(EPOCHS):
    model.train()
    total_loss   = 0
    total_images = 0

    torch.cuda.synchronize()
    ep_start = time.perf_counter()

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        total_images += x.size(0)

    torch.cuda.synchronize()
    ep_time    = time.perf_counter() - ep_start
    throughput = total_images / ep_time
    avg_loss   = total_loss / len(train_loader)

    val_m = evaluate(val_loader)

    tflops_epoch = (
        (total_flops * len(train_loader) * 3) / 1e12
        if total_flops else float("nan")
    )

    epoch_metrics.append({
        "epoch"  : epoch + 1,
        "loss"   : avg_loss,
        "val_acc": val_m["accuracy"],
        "val_f1" : val_m["f1"],
        "time_s" : ep_time,
        "img_s"  : throughput,
        "tflops" : tflops_epoch,
    })

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss {avg_loss:.4f} | "
        f"Val Acc {val_m['accuracy']:.4f} | "
        f"Val F1 {val_m['f1']:.4f} | "
        f"Time {ep_time:.1f}s | "
        f"Throughput {throughput:.1f} img/s | "
        f"TFLOPs {tflops_epoch:.2f}"
    )

total_time = time.perf_counter() - total_start

# ─────────────────────────────
# TEST EVALUATION
# ─────────────────────────────
test_m = evaluate(test_loader)

# ─────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────
print("\n" + "="*70)
print("          STREAM FUSION — FINAL METRICS SUMMARY")
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
print(f"  {'Ep':<4} {'Loss':<9} {'ValAcc':<9} {'ValF1':<9} {'Time(s)':<9} {'Img/s':<9} {'TFLOPs'}")
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
print(f"  Accuracy  : {test_m['accuracy']:.4f} ({test_m['accuracy']*100:.2f}%)")
print(f"  Precision : {test_m['precision']:.4f}")
print(f"  Recall    : {test_m['recall']:.4f}")
print(f"  F1 Score  : {test_m['f1']:.4f}")

print(f"\n[Confusion Matrix]")
print(test_m["confusion"])

# ─────────────────────────────
# PDC COMPARISON
# ─────────────────────────────
SEQ_TIME       = 2561.77   # your measured sequential time
SEQ_THROUGHPUT = 227.2    # your measured sequential throughput
NUM_STREAMS    = 3

speedup    = SEQ_TIME / total_time
efficiency = (speedup / NUM_STREAMS) * 100

print(f"\n[PDC Comparison]")
print(f"  Sequential time     : {SEQ_TIME:.2f}s")
print(f"  Stream parallel time: {total_time:.2f}s")
print(f"  Speedup (S)         : {speedup:.3f}x")
print(f"  Efficiency (E)      : {efficiency:.2f}%")
print(f"  Sequential tput     : {SEQ_THROUGHPUT:.1f} img/s")
print(f"  Stream tput         : {avg_throughput:.1f} img/s")
print(f"  Throughput gain     : {avg_throughput/SEQ_THROUGHPUT:.3f}x")
print("="*70)