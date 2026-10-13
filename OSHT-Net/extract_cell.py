"""Extract cell-morphological features (Sec. 3.3, "Cell-Morphological Features").

Pipeline per ROI:
    1. Cellpose nuclei instance segmentation (<= 3000 nuclei kept)
    2. 64x64 microenvironment crop around each centroid (reflection padding)
    3. encode each crop (resized to 224x224) with the UNI ViT-Large pathology
       foundation model -> 1024-d features saved as ``<roi>_cell.npy``

Usage:
    python extract_cell.py --split train
"""
import fix_startup  # noqa: F401
import argparse
import glob
import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from skimage.measure import regionprops
from torchvision import transforms

import config

CROP = 64
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def init_cellpose(weights_path):
    from cellpose import models
    model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=weights_path)
    return model


def load_uni(weights_path):
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
                              init_values=1e-5, num_classes=0)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    model.load_state_dict(state, strict=False)
    model.cuda().eval()
    return model


def cell_crops_from_image(img_rgb, cellpose_model, crop_size=64, max_cells=3000):
    masks, flows, styles = cellpose_model.eval(img_rgb, diameter=None, channels=[0, 0])
    h, w = img_rgb.shape[:2]
    centroids = [(int(p.centroid[1]), int(p.centroid[0])) for p in regionprops(masks)]
    if len(centroids) > max_cells:
        centroids = random.sample(centroids, max_cells)

    half = crop_size // 2
    patches = []
    for cx, cy in centroids:
        x1, x2 = cx - half, cx + half
        y1, y2 = cy - half, cy + half
        pad_left, pad_top = max(0, -x1), max(0, -y1)
        pad_right, pad_bottom = max(0, x2 - w), max(0, y2 - h)
        crop = img_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if pad_left or pad_top or pad_right or pad_bottom:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_REFLECT_101)
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            crop = cv2.resize(crop, (crop_size, crop_size))
        patches.append(Image.fromarray(crop))
    return patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_root = os.path.join(config.FEATURE_ROOT, "cell_feature")
    cellpose_model = init_cellpose(config.CELLPOSE_WEIGHTS)
    uni_model = load_uni(config.UNI_WEIGHTS)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    split_dir = os.path.join(config.BRACS_ROOT, args.split)
    img_paths = sorted(glob.glob(os.path.join(split_dir, "**", "*.png"), recursive=True))
    for path in img_paths:
        rel = os.path.relpath(path, config.BRACS_ROOT)
        parent = os.path.dirname(rel)
        base = os.path.basename(path).replace(".png", "")
        out_file = os.path.join(out_root, parent, f"{base}_cell.npy")
        if os.path.exists(out_file):
            continue
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        try:
            patches = cell_crops_from_image(img_rgb, cellpose_model,
                                            CROP, config.FEATURE_CFG["cell"]["max_cells"])
        except Exception as e:
            print(f"[cell/{args.split}] {parent}/{base}: segmentation failed ({e}), skip")
            continue
        if not patches:
            continue

        tensors = torch.stack([transform(p) for p in patches])
        feats = []
        with torch.no_grad():
            for i in range(0, tensors.size(0), args.batch_size):
                feats.append(uni_model(tensors[i:i + args.batch_size].cuda()).cpu().numpy())
        np.save(out_file, np.vstack(feats).astype(np.float32))
        print(f"[cell/{args.split}] {parent}/{base}: {len(patches)} cells")

    print("done.")


if __name__ == "__main__":
    main()
