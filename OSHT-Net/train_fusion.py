"""Train the Hierarchical TransMIL fusion network on precomputed features.

The four feature modalities (comp / ssl / edge / cell) are precomputed by
``extract_overlock.py``, ``extract_edge.py`` and ``extract_cell.py``. This
script trains only the fusion network with ROI-level labels (Sec. 3.5, Table V(b)).

Usage:
    python train_fusion.py                          # all four modalities
    python train_fusion.py --features comp edge     # ablation (optional)
"""
import fix_startup  # noqa: F401
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from models import HierarchicalTransMIL

FEATURE_DIRS = {
    "comp": os.path.join(config.FEATURE_ROOT, "overlock_feature"),
    "ssl": os.path.join(config.FEATURE_ROOT, "overlock_feature"),
    "edge": os.path.join(config.FEATURE_ROOT, "edge_feature"),
    "cell": os.path.join(config.FEATURE_ROOT, "cell_feature"),
}
FEATURE_SUFFIXES = {"comp": ".npy", "ssl": "_ssl.npy", "edge": "_edge.npy", "cell": "_cell.npy"}
FEATURE_DIMS = {"comp": 1024, "ssl": 1024, "edge": 384, "cell": 1024}


class MultiModalDataset(Dataset):
    """Loads aligned feature files for a list of modalities."""

    def __init__(self, active_features, phase="train"):
        self.active_features = active_features
        self.samples = []
        base = active_features[0]
        suffix = FEATURE_SUFFIXES[base]
        pattern = os.path.join(FEATURE_DIRS[base], phase, "**", f"*{suffix}")
        # Some modalities share one directory (comp "<roi>.npy" and ssl
        # "<roi>_ssl.npy" both live in overlock_feature/). A file that ends with
        # a *more specific* suffix of another modality belongs to that modality
        # and must not be collected as a ``base`` sample.
        other_suffixes = [
            s for f, s in FEATURE_SUFFIXES.items()
            if f != base and s.endswith(suffix) and len(s) > len(suffix)
        ]
        for file_path in glob.glob(pattern, recursive=True):
            filename = os.path.basename(file_path)
            if any(filename.endswith(s) for s in other_suffixes):
                continue
            cls_dir = os.path.basename(os.path.dirname(file_path))
            basename = filename[: -len(suffix)]
            try:
                label = int(cls_dir.split("_")[0])
            except ValueError:
                continue
            paths = {}
            for feat in active_features:
                p = os.path.join(FEATURE_DIRS[feat], phase, cls_dir,
                                 f"{basename}{FEATURE_SUFFIXES[feat]}")
                if not os.path.exists(p):
                    break
                paths[feat] = p
            else:
                self.samples.append({"paths": paths, "label": label, "basename": basename})
        print(f"[{phase}] {len(self.samples)} samples for {active_features}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tensors = {}
        for feat, path in sample["paths"].items():
            arr = np.load(path, allow_pickle=True)
            if arr.dtype == object and arr.ndim == 0:
                arr = arr.item()
            if isinstance(arr, dict):
                arr = arr.get("features", arr.get("feat", list(arr.values())[0]))
            if isinstance(arr, (list, tuple)):
                arr = np.vstack(arr)
            tensors[feat] = torch.from_numpy(np.array(arr).astype(np.float32))
        return tensors, torch.tensor(sample["label"], dtype=torch.long)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    for batch_feats, batch_labels in dataloader:
        batch_labels = batch_labels.to(device)
        logits = model(batch_feats)
        loss = criterion(logits, batch_labels)
        total_loss += loss.item()
        total += batch_labels.size(0)
        pred = logits.argmax(1)
        correct += (pred == batch_labels).sum().item()
        preds.extend(pred.cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
    acc = correct / total
    wf1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return total_loss / len(dataloader), acc, wf1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+",
                        default=config.FUSION_CFG["active_features"])
    parser.add_argument("--hidden-dim", type=int, default=config.FUSION_CFG["hidden_dim"])
    parser.add_argument("--dropout", type=float, default=config.FUSION_CFG["dropout"])
    parser.add_argument("--lr", type=float, default=config.FUSION_CFG["lr"])
    parser.add_argument("--epochs", type=int, default=config.FUSION_CFG["epochs"])
    parser.add_argument("--output", default="output_fusion")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    train_ds = MultiModalDataset(args.features, "train")
    val_ds = MultiModalDataset(args.features, "val")
    test_ds = MultiModalDataset(args.features, "test")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    model = HierarchicalTransMIL(
        {f: FEATURE_DIMS[f] for f in args.features},
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        num_classes=config.FUSION_CFG["num_classes"]).to(device)
    print(f"Fusion model params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=config.FUSION_CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.FUSION_CFG["lr_factor"],
        patience=config.FUSION_CFG["lr_patience"])

    best_val_f1, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_feats, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                                              leave=False):
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_feats)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += batch_labels.size(0)
            train_correct += (logits.argmax(1) == batch_labels).sum().item()

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.output, "model_best.pth"))

        history.append(dict(epoch=epoch, train_loss=train_loss / len(train_loader),
                            train_acc=train_correct / train_total, val_loss=val_loss,
                            val_acc=val_acc, val_f1=val_f1, test_acc=test_acc, test_f1=test_f1))
        print(f"Epoch {epoch:03d} | train_loss={train_loss / len(train_loader):.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | test_acc={test_acc:.4f} "
              f"test_f1={test_f1:.4f} | best_val_f1={best_val_f1:.4f}")

    pd.DataFrame(history).to_csv(os.path.join(args.output, "metrics.csv"), index=False)
    print(f"Best val W-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
