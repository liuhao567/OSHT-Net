"""Extract comp / ssl features with the OverLoCK-SCSA backbone.

The backbone is applied to 224x224 grid patches of the original-resolution ROI
(no resizing). The 1024-d embedding before the linear classifier is saved for
every patch together with its top-left coordinate.

Usage:
    python extract_overlock.py --mode comp --split train
    python extract_overlock.py --mode ssl  --split train
"""
import fix_startup  # noqa: F401  (workaround for a Python 3.10 regex bug)
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from timm.models import create_model
from torchvision import transforms

import config
import models  # noqa: F401  (registers overlock_xt_bridge_scsa)

PATCH = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_backbone(ckpt: str, num_classes: int):
    """Build the backbone and load a checkpoint.

    ``num_classes`` must match the checkpoint: the supervised backbone was
    trained with ``num_classes=7`` (it contains ``head.4.*``), while the MAE
    encoder was saved with ``num_classes=0`` (no ``head.4.*`` parameters).
    """
    model = create_model("overlock_xt_bridge_scsa", pretrained=False,
                         num_classes=num_classes)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    return model


@torch.no_grad()
def extract_roi(model, img_path: str, transform):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    pad_w = (PATCH - W % PATCH) % PATCH
    pad_h = (PATCH - H % PATCH) % PATCH
    img = transforms.functional.pad(img, (0, 0, pad_w, pad_h), padding_mode="reflect")
    W, H = img.size

    feats, coords = [], []
    for y in range(0, H, PATCH):
        for x in range(0, W, PATCH):
            crop = img.crop((x, y, x + PATCH, y + PATCH))
            inp = transform(crop).unsqueeze(0).cuda()
            feat, _ = model.forward_features(inp)
            emb = model.head[:4](feat).flatten(1)   # 1024-d embedding
            feats.append(emb.cpu().numpy())
            coords.append((x, y))
    return np.vstack(feats).astype(np.float32), np.array(coords, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["comp", "ssl"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    ckpt = args.ckpt or (config.BACKBONE_CKP if args.mode == "comp" else config.SSL_CKP)
    suffix = ".npy" if args.mode == "comp" else "_ssl.npy"
    out_root = os.path.join(config.FEATURE_ROOT, "overlock_feature")

    # comp: supervised backbone trained with 7 classes.
    # ssl: MAE encoder saved without the classification head (num_classes=0).
    num_classes = 7 if args.mode == "comp" else 0
    model = load_backbone(ckpt, num_classes)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    split_dir = os.path.join(config.BRACS_ROOT, args.split)
    for cls_dir in sorted(os.listdir(split_dir)):
        cls_path = os.path.join(split_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        out_cls = os.path.join(out_root, args.split, cls_dir)
        os.makedirs(out_cls, exist_ok=True)
        for img_path in sorted(glob.glob(os.path.join(cls_path, "*.png"))):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            out_file = os.path.join(out_cls, basename + suffix)
            if os.path.exists(out_file):
                continue
            feats, coords = extract_roi(model, img_path, transform)
            np.save(out_file, {"features": feats, "coords": coords}, allow_pickle=True)
            print(f"[{args.mode}/{args.split}] {cls_dir}/{basename}: {feats.shape}")

    print("done.")


if __name__ == "__main__":
    main()
