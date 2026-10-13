"""Evaluate a trained Hierarchical TransMIL fusion model.

Reports the four metrics used in the paper (Balanced Accuracy, Weighted F1,
Macro Precision, Macro AUC) plus accuracy, for the 7-class and/or 3-class
setting on the validation or test split.

Usage:
    python evaluate.py --split test --ckpt output_fusion/model_best.pth
    python evaluate.py --split test --ckpt output_fusion/model_best.pth --three-class
"""
import fix_startup  # noqa: F401
import argparse
import glob
import os

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader

import config
from models import HierarchicalTransMIL
from train_fusion import FEATURE_DIRS, FEATURE_DIMS, FEATURE_SUFFIXES, MultiModalDataset


@torch.no_grad()
def evaluate(model, dataloader, device, three_class=False):
    model.eval()
    preds, labels, probs = [], [], []
    for batch_feats, batch_labels in dataloader:
        batch_labels = batch_labels.to(device)
        logits = model(batch_feats)
        prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
        if three_class:
            batch_labels = batch_labels.cpu().numpy()
            pred = np.vectorize(config.CLASS_GROUPS_3.get)(pred)
            batch_labels = np.vectorize(config.CLASS_GROUPS_3.get)(batch_labels)
            # re-normalize 7-class probabilities into 3 groups
            prob3 = np.zeros((prob.shape[0], 3), dtype=prob.dtype)
            for c, g in config.CLASS_GROUPS_3.items():
                prob3[:, g] += prob[:, c]
            prob = prob3
        preds.append(pred)
        labels.append(batch_labels)
        probs.append(prob)

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)

    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    w_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    m_prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    m_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") * 100
    return {"Acc": acc, "Bal.Acc": bal_acc, "W-F1": w_f1, "M-Prec": m_prec, "M-AUC": m_auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--features", nargs="+",
                        default=config.FUSION_CFG["active_features"])
    parser.add_argument("--hidden-dim", type=int, default=config.FUSION_CFG["hidden_dim"])
    parser.add_argument("--dropout", type=float, default=config.FUSION_CFG["dropout"])
    parser.add_argument("--three-class", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalTransMIL(
        {f: FEATURE_DIMS[f] for f in args.features},
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        num_classes=config.FUSION_CFG["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))

    ds = MultiModalDataset(args.features, args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    metrics = evaluate(model, loader, device, three_class=args.three_class)
    tag = "3-class" if args.three_class else "7-class"
    print(f"[{tag} / {args.split}] " + "  ".join(f"{k}={v:.2f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
