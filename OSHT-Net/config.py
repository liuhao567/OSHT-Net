"""Central configuration for OSHT-Net.

Edit the paths below before running any script. All other scripts import this
module so that no path is hard-coded anywhere else.

Paths marked with ``/path/to/...`` are placeholders: replace them with the
locations on your machine (or leave ``None`` where automatic download is
supported).
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

# BRACS ROI images. Expected layout (one folder per class, class id first):
#   <BRACS_ROOT>/train/<class_id>_<class_name>/*.png
#   <BRACS_ROOT>/val/<class_id>_<class_name>/*.png
#   <BRACS_ROOT>/test/<class_id>_<class_name>/*.png
# Example class folder names: 0_N, 1_PB, 2_UDH, 3_FEA, 4_ADH, 5_DCIS, 6_IC.
BRACS_ROOT = "/path/to/BRACS"

# Root folder where extracted features (comp/ssl/edge/cell) are written.
# Any writable location is fine; it will contain overlock_feature/,
# edge_feature/ and cell_feature/ automatically.
FEATURE_ROOT = "/path/to/oshtnet_features"

# ImageNet-1K checkpoint of the plain OverLoCK-XT.
# Download from https://github.com/LMMMEng/OverLoCK/releases and put it under
# <repo>/pretrained/overlock_xt_in1k_224.pth
PRETRAINED_OVERLOCK = os.path.join(ROOT, "pretrained", "overlock_xt_in1k_224.pth")

# Checkpoints produced by the training scripts in this repo:
#   BACKBONE_CKP <- train.py (Step 1, comp extractor)
#   SSL_CKP      <- pretrain_mae.py (Step 2, ssl extractor)
BACKBONE_CKP = os.path.join(ROOT, "output", "backbone_bridge_scsa",
                            "checkpoints", "model_best.pth.tar")
SSL_CKP = os.path.join(ROOT, "ssl_ckp", "overlock_xt_bridge_mae_best.pth")

# UNI foundation model weights (cell branch).
# Download pytorch_model.bin from https://huggingface.co/MahmoodLab/UNI
UNI_WEIGHTS = "/path/to/UNI/pytorch_model.bin"

# Cellpose nuclei model (cell branch).
#   None  -> use Cellpose's built-in "nuclei" model (auto-download on first use)
#   str   -> path to a local model file, e.g. "/path/to/nucleitorch_0"
CELLPOSE_WEIGHTS = None

# ---------------------------------------------------------------------------
# Backbone training (Table V(a) in the paper)
# ---------------------------------------------------------------------------
BACKBONE_CFG = dict(
    input_size=224,
    batch_size=8,
    lr=1e-3,
    auto_lr=True,
    drop_path=0.4,
    epochs=250,
    warmup_epochs=5,
    workers=8,
    native_amp=True,
    clip_grad=5.0,
    num_classes=7,
    weight_decay=0.05,
    optimizer="adamw",
)

# ---------------------------------------------------------------------------
# MAE self-supervised pretraining (ssl branch)
# ---------------------------------------------------------------------------
MAE_CFG = dict(
    img_size=224,
    patch_size=16,
    mask_ratio=0.6,
    batch_size=16,
    epochs=150,
    base_lr=1.5e-4,
    weight_decay=0.05,
    num_workers=8,
)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
FEATURE_CFG = dict(
    comp=dict(extractor="overlock_scsa", ckp=BACKBONE_CKP, dim=1024, patch=224),
    ssl=dict(extractor="overlock_scsa", ckp=SSL_CKP, dim=1024, patch=224),
    edge=dict(extractor="vit_small", dim=384, crop=64, stride=32,
              min_contour_area=500, morph_kernel=15),
    cell=dict(extractor="uni", dim=1024, crop=64, max_cells=3000),
)

# ---------------------------------------------------------------------------
# Hierarchical TransMIL fusion (Table V(b) in the paper)
# ---------------------------------------------------------------------------
FUSION_CFG = dict(
    hidden_dim=8,
    dropout=0.2,
    nhead=4,
    fusion_layers=2,
    batch_size=1,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=30,
    lr_patience=5,
    lr_factor=1 / 3,
    num_classes=7,
    active_features=["comp", "ssl", "edge", "cell"],
)

# 3-class merge (used for the 3-class evaluation in the paper)
CLASS_GROUPS_3 = {
    0: 0,  # N      -> Benign
    1: 0,  # PB     -> Benign
    2: 1,  # UDH    -> Atypical
    3: 1,  # FEA    -> Atypical
    4: 1,  # ADH    -> Atypical
    5: 2,  # DCIS   -> Malignant
    6: 2,  # IC     -> Malignant
}
