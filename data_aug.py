"""
Assignment 2 – Problem 5
Data Augmentation Experiment with LeNet-5 on ReducedMNIST
==========================================================
Grid: 3 real-data sizes × 4 augmentation levels = 12 experiments
Real sizes per digit  : 350 / 750 / 1000
Generated per digit   : 0 / 1000 / 1500 / 2000
Test set (fixed)      : 200 per digit = 2000 total
"""

import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from skimage.feature import hog

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 28
NUM_EPOCHS = 15
BATCH_SIZE = 64
LR         = 0.001

print(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – LOAD & PREPARE MNIST
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading MNIST...")
train_full = datasets.MNIST(root='./data', train=True,  download=True)
test_full  = datasets.MNIST(root='./data', train=False, download=True)

train_images_all = train_full.data.numpy()          # (60000, 28, 28)  uint8
train_labels_all = train_full.targets.numpy()
test_images_all  = test_full.data.numpy()
test_labels_all  = test_full.targets.numpy()


def balanced_subsample(images, labels, n_per_class):
    """Return a balanced subset with exactly n_per_class samples per digit."""
    imgs, lbls = [], []
    for digit in range(10):
        idx = np.where(labels == digit)[0]
        chosen = np.random.choice(idx, n_per_class, replace=False)
        imgs.append(images[chosen])
        lbls.append(labels[chosen])
    return np.vstack(imgs), np.hstack(lbls)


# Fixed test set: 200 per digit
test_images, test_labels = balanced_subsample(test_images_all, test_labels_all, 200)
test_images = test_images / 255.0                   # normalise → [0, 1]

print(f"   Test set : {test_images.shape}  labels {test_labels.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def augment_images(images_norm, labels, n_generate_per_digit):
    """
    Given normalised images [0,1] and their labels, generate
    n_generate_per_digit augmented copies per digit class.

    Augmentation operations applied randomly per sample:
      • Rotation   : uniform ±15°
      • Translation: uniform ±3 px in x and y
      • White noise: 3 % of pixel range added
      • Zoom       : random scale 0.9 – 1.1 (keeps output 28×28)

    Returns (aug_images, aug_labels) as float32 arrays, normalised [0,1].
    """
    if n_generate_per_digit == 0:
        return np.array([]).reshape(0, IMG_SIZE, IMG_SIZE), np.array([], dtype=np.int64)

    aug_imgs, aug_lbls = [], []

    for digit in range(10):
        digit_mask  = labels == digit
        digit_imgs  = images_norm[digit_mask]          # (n_real, 28, 28)
        n_real      = len(digit_imgs)

        generated = 0
        while generated < n_generate_per_digit:
            # pick a random source image for this digit
            src = digit_imgs[np.random.randint(n_real)].copy()
            img = (src * 255).astype(np.uint8)         # cv2 works on uint8

            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

            # --- rotation ---
            angle = np.random.uniform(-15, 15)
            M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            img   = cv2.warpAffine(img, M_rot, (IMG_SIZE, IMG_SIZE),
                                   borderMode=cv2.BORDER_REPLICATE)

            # --- translation ---
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            M_tr = np.float32([[1, 0, dx], [0, 1, dy]])
            img  = cv2.warpAffine(img, M_tr, (IMG_SIZE, IMG_SIZE),
                                  borderMode=cv2.BORDER_REPLICATE)

            # --- zoom (scale while keeping 28×28) ---
            scale = np.random.uniform(0.9, 1.1)
            M_sc  = cv2.getRotationMatrix2D((cx, cy), 0, scale)
            img   = cv2.warpAffine(img, M_sc, (IMG_SIZE, IMG_SIZE),
                                   borderMode=cv2.BORDER_REPLICATE)

            # --- white noise ---
            noise = np.random.normal(0, 0.03 * 255, img.shape).astype(np.float32)
            img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            aug_imgs.append(img.astype(np.float32) / 255.0)
            aug_lbls.append(digit)
            generated += 1

    return np.array(aug_imgs, dtype=np.float32), np.array(aug_lbls, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – LeNet-5  (adapted for 28 × 28 input, ReLU activations)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Input  28×28×1
#  Conv1  5×5, 6 filters, stride 1  → 24×24×6
#  ReLU
#  AvgPool 2×2, stride 2            → 12×12×6
#  Conv2  5×5, 16 filters, stride 1 → 8×8×16
#  ReLU
#  AvgPool 2×2, stride 2            → 4×4×16  = 256 flat
#  FC     256 → 120 → 84 → 10
#  Softmax (cross-entropy loss handles this internally)

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6,  kernel_size=5),   # 28→24
            nn.ReLU(),
            nn.AvgPool2d(2, 2),                # 24→12
            nn.Conv2d(6, 16, kernel_size=5),   # 12→8
            nn.ReLU(),
            nn.AvgPool2d(2, 2),                # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – TRAIN / EVAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_loader(images_np, labels_np, shuffle=True):
    """Convert numpy arrays → PyTorch DataLoader (adds channel dim)."""
    X = torch.tensor(images_np, dtype=torch.float32).unsqueeze(1)  # (N,1,28,28)
    y = torch.tensor(labels_np, dtype=torch.long)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(train_loader, label=""):
    """Train LeNet-5 for NUM_EPOCHS. Returns (model, training_time_ms)."""
    model     = LeNet5().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        # ── training pass ──────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (out.argmax(dim=1) == yb).sum().item()
            total      += yb.size(0)

        train_acc  = correct / total * 100
        train_loss = total_loss / len(train_loader)

        # ── test pass (only from epoch 15 onward) ──────────────────
        if epoch >= 0:
            model.eval()
            t_correct, t_total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    t_correct += (model(xb).argmax(dim=1) == yb).sum().item()
                    t_total   += yb.size(0)
            test_acc = t_correct / t_total * 100
            print(f"     Epoch {epoch:>2}/{NUM_EPOCHS}  "
                  f"loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
                  f"test_acc={test_acc:.1f}%")
        else:
            print(f"     Epoch {epoch:>2}/{NUM_EPOCHS}  "
                  f"loss={train_loss:.4f}  train_acc={train_acc:.1f}%")

    train_time_ms = (time.time() - t0) * 1000
    return model, train_time_ms


def evaluate_model(model, test_loader):
    """Return (accuracy_%, testing_time_ms)."""
    model.eval()
    correct, total = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds   = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    test_time_ms = (time.time() - t0) * 1000
    accuracy = round(correct / total * 100, 1)
    return accuracy, round(test_time_ms, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – EXPERIMENT GRID
# ═══════════════════════════════════════════════════════════════════════════════

REAL_SIZES   = [350, 750, 1000]       # columns
GEN_SIZES    = [0, 1000, 1500, 2000]  # rows

# results[gen_idx][real_idx] = {"acc": %, "train_ms": ms, "test_ms": ms}
results = [[None]*len(REAL_SIZES) for _ in range(len(GEN_SIZES))]

# Build fixed test loader once
test_loader = make_loader(test_images, test_labels, shuffle=False)

print("\n[2] Starting experiment grid  (3 real sizes × 4 augmentation levels) …")
print("=" * 70)

for ri, n_real in enumerate(REAL_SIZES):
    # sample the real training set for this column
    real_imgs, real_lbls = balanced_subsample(
        train_images_all, train_labels_all, n_real
    )
    real_imgs = real_imgs / 255.0

    for gi, n_gen in enumerate(GEN_SIZES):

        label = (f"  Real/digit={n_real:>4}  |  "
                 f"Generated/digit={n_gen:>4}  |  "
                 f"Total/digit≈{n_real + n_gen:>4}")
        print(label)

        # --- build augmented portion ---
        aug_imgs, aug_lbls = augment_images(real_imgs, real_lbls, n_gen)
        # --- combine real + augmented ---
        if n_gen > 0:
            all_imgs = np.concatenate([real_imgs, aug_imgs], axis=0)
            all_lbls = np.concatenate([real_lbls, aug_lbls], axis=0)
        else:
            all_imgs = real_imgs
            all_lbls = real_lbls

        print(f"     Training samples: {len(all_imgs)}")

        # --- shuffle combined set ---
        perm     = np.random.permutation(len(all_imgs))
        all_imgs = all_imgs[perm]
        all_lbls = all_lbls[perm]

        train_loader = make_loader(all_imgs, all_lbls, shuffle=True)

        # --- train & evaluate ---
        model, train_ms = train_model(train_loader, label=label)
        acc, test_ms    = evaluate_model(model, test_loader)

        results[gi][ri] = {
            "acc":      acc,
            "train_ms": round(train_ms, 1),
            "test_ms":  test_ms,
        }

        print(f"     → Accuracy: {acc}%   Train: {train_ms:.1f} ms   "
              f"Test: {test_ms:.1f} ms\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 – PRINT RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESULTS TABLE – Accuracy (%) / Training time (ms) / Testing time (ms)")
print("=" * 70)

# header
print(f"\n{'Generated/digit':<18} | {'350 real/digit':^26} | "
      f"{'750 real/digit':^26} | {'1000 real/digit':^26}")
print("-" * 103)

for gi, n_gen in enumerate(GEN_SIZES):
    label = f"{n_gen}"
    row_parts = []
    for ri in range(len(REAL_SIZES)):
        r = results[gi][ri]
        cell = f"{r['acc']}%  {r['train_ms']} ms  {r['test_ms']} ms"
        row_parts.append(f"{cell:^26}")
    print(f"{label:<18} | " + " | ".join(row_parts))

print()

# ── compact accuracy-only table ──────────────────────────────────────────────
print("ACCURACY SUMMARY TABLE (%)")
print(f"\n{'Gen/digit':<12} | {'350 real':^12} | {'750 real':^12} | {'1000 real':^12}")
print("-" * 55)
for gi, n_gen in enumerate(GEN_SIZES):
    accs = [str(results[gi][ri]['acc']) for ri in range(len(REAL_SIZES))]
    print(f"{n_gen:<12} | {accs[0]:^12} | {accs[1]:^12} | {accs[2]:^12}")

print("\nDone ✓")