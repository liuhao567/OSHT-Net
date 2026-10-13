"""Extract edge-structural features (Sec. 3.3, "Edge-Structural Features").

Pipeline per ROI:
    1. grayscale -> 5x5 Gaussian blur -> Otsu threshold
    2. 15x15 morphological closing -> external contours (area >= 500 px)
    3. sample a 64x64 crop every 32 px along each contour (reflection padding)
    4. encode each crop (resized to 224x224) with ImageNet-pretrained ViT-Small
       -> 384-d features saved as ``<roi>_edge.npy``

Usage:
    python extract_edge.py --split train
"""
import fix_startup  # noqa: F401
import argparse
import glob
import os

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

import config

CROP, STRIDE = 64, 32
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def edge_crops_from_image(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    h, w = img_rgb.shape[:2]
    patches = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        contour = contour.squeeze(1)
        for i in range(0, len(contour), STRIDE):
            cx, cy = contour[i]
            x1, x2 = cx - CROP // 2, cx + CROP // 2
            y1, y2 = cy - CROP // 2, cy + CROP // 2

            pad_left, pad_top = max(0, -x1), max(0, -y1)
            pad_right, pad_bottom = max(0, x2 - w), max(0, y2 - h)
            crop = img_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if pad_left or pad_top or pad_right or pad_bottom:
                crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_REFLECT_101)
            patches.append(Image.fromarray(crop))
    return patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    out_root = os.path.join(config.FEATURE_ROOT, "edge_feature")
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    model.cuda().eval()

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
        out_file = os.path.join(out_root, parent, f"{base}_edge.npy")
        if os.path.exists(out_file):
            continue
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        patches = edge_crops_from_image(img_rgb)
        if not patches:
            continue

        tensors = torch.stack([transform(p) for p in patches])
        feats = []
        with torch.no_grad():
            for i in range(0, tensors.size(0), args.batch_size):
                feats.append(model(tensors[i:i + args.batch_size].cuda()).cpu().numpy())
        np.save(out_file, np.vstack(feats).astype(np.float32))
        print(f"[edge/{args.split}] {parent}/{base}: {len(patches)} patches")

    print("done.")


if __name__ == "__main__":
    main()
