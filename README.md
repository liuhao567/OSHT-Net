# OSHT-Net

**OSHT-Net: An OverLoCK-SCSA Backbone with a Hierarchical TransMIL Fusion Network for Breast Cancer Histopathological Image Classification**

Official PyTorch implementation.

<!-- 图1：总体架构图（论文 Fig.1）放到这里 —— 保存成 assets/fig1.png -->
<p align="center"><img src="assets/fig1.png" width="100%"></p>

OSHT-Net classifies a breast histopathology ROI into seven subtypes (N, PB, UDH, FEA, ADH, DCIS, IC). One ROI is mapped to four complementary feature sequences: **comp** (tissue texture, OverLoCK-SCSA supervised on BRACS), **ssl** (self-supervised semantics, MAE-pretrained), **edge** (boundary structure, ViT-Small), and **cell** (nuclear morphology, Cellpose + UNI). **Hierarchical TransMIL** first aggregates each sequence into a modality token, then fuses the four tokens for the final seven-class prediction. The backbone follows the “overview-first, look-closely-next” design, enhanced with **SCSA** spatial-channel attention and a **Context Bridge-SCSA** module at the entry of each Focus-Net sub-stage.

<!-- 图2：四模态特征提取 + 分层融合框架图放到这里 —— 保存成 assets/fig2.png -->
<p align="center"><img src="assets/fig2.png" width="100%"></p>

## Pipeline

| Modality | Extractor | Cropping | Dim |
|---|---|---|---|
| `comp` | OverLoCK-SCSA (BRACS-supervised) | 224×224 grid patches | 1024 |
| `ssl` | OverLoCK-SCSA (MAE-pretrained) | 224×224 grid patches | 1024 |
| `edge` | ViT-Small (ImageNet) | 64×64 boundary patches | 384 |
| `cell` | UNI ViT-Large | 64×64 nucleus-centered patches | 1024 |

## Results (BRACS validation, %)

| Method | 7-cls Bal.Acc | 7-cls W-F1 | 7-cls M-Prec | 7-cls M-AUC | 3-cls Bal.Acc | 3-cls W-F1 | 3-cls M-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-V2 | 53.77 | 53.75 | 54.71 | 86.18 | 73.78 | 73.02 | 89.64 |
| DenseNet121 | 47.54 | 46.89 | 48.23 | 86.23 | 72.98 | 72.71 | 89.99 |
| TransMIL | 53.15 | 53.45 | 54.49 | 86.79 | 65.43 | 66.77 | 85.92 |
| HactNet | 53.99 | 53.40 | 54.04 | 85.76 | 64.06 | 65.58 | 84.77 |
| ScoreNet | 64.67 | 64.53 | 65.06 | 91.27 | 80.76 | 80.27 | 93.08 |
| CLAM | 64.38 | 65.52 | 65.00 | **91.90** | 77.72 | 77.81 | 92.24 |
| **OSHT-Net (Ours)** | **65.51** | **66.10** | **67.86** | **91.90** | **86.32** | **85.63** | **94.99** |

Backbone-only Bridge-SCSA variant: **70.60 Bal.Acc / 69.78 W-F1 / 70.61 M-Prec / 92.98 M-AUC**. Fine-tuning on BACH: **83.3%** accuracy.

## Repository structure

```
OSHT-Net/
├── models/
│   ├── SCSA.py                  # SCSA attention + SCSA Residual Block
│   ├── overlock_scsa.py         # OverLoCK-SCSA backbone (Context Bridge SCSA)
│   └── hierarchical_transmil.py # single-stream + global fusion Transformers
├── config.py                    # all paths & hyper-parameters (edit this first)
├── train.py                     # backbone training on BRACS
├── pretrain_mae.py              # MAE self-supervised pretraining (ssl branch)
├── extract_overlock.py          # comp / ssl feature extraction
├── extract_edge.py              # edge feature extraction
├── extract_cell.py              # cell feature extraction (Cellpose + UNI)
├── train_fusion.py              # Hierarchical TransMIL fusion training
├── evaluate.py                  # 7-class / 3-class evaluation
└── pretrained/                  # put overlock_xt_in1k_224.pth here
```

## Installation

Tested with Python 3.10 / CUDA 12.1 / a single RTX 4090.

```bash
conda create -n oshtnet python=3.10 -y && conda activate oshtnet
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
pip install -r requirements.txt
```

## Data and weights

Download [BRACS](https://www.bracs.icar.cnr.it/) and organize the ROIs as `BRACS/{train,val,test}/<class_id>_<name>/*.png` (class folders `0_N` … `6_IC`, split 3,657/312/570).

| Weight | Used by | Where to put it |
|---|---|---|
| `overlock_xt_in1k_224.pth` | backbone ImageNet init | `pretrained/` ([OverLoCK releases](https://github.com/LMMMEng/OverLoCK/releases)) |
| `model_best.pth.tar` (backbone) | comp extractor | `output/backbone_bridge_scsa/checkpoints/` |
| `overlock_xt_bridge_mae_best.pth` | ssl extractor | `ssl_ckp/` |
| `pytorch_model.bin` (UNI) | cell extractor | set `UNI_WEIGHTS` in `config.py` |
| Cellpose nuclei model | cell segmentation | auto-download, or set `CELLPOSE_WEIGHTS` |
| `model_best.pth` (fusion) | evaluation | `output_fusion/` |

Edit `BRACS_ROOT`, `FEATURE_ROOT`, `UNI_WEIGHTS` in `config.py` before running anything.

## Reproduction

```bash
# 1. OverLoCK-SCSA backbone (comp branch)
python train.py --data-dir $BRACS_ROOT --train-split train --val-split val \
  --batch-size 8 --model overlock_xt_bridge_scsa \
  --lr 1e-3 --auto-lr --drop-path 0.4 --epochs 250 --warmup-epochs 5 \
  --workers 8 --native-amp --clip-grad 5 --num-classes 7 \
  --checkpoint-hist 2 --early-stop-patience 30 \
  --init-with-ckp pretrained/overlock_xt_in1k_224.pth \
  --output output --experiment backbone_bridge_scsa

# 2. MAE pretraining (ssl branch)
python pretrain_mae.py

# 3. Extract the four feature modalities
for split in train val test; do
  python extract_overlock.py --mode comp --split $split
  python extract_overlock.py --mode ssl  --split $split
  python extract_edge.py --split $split
  python extract_cell.py --split $split
done

# 4. Train the Hierarchical TransMIL fusion network
python train_fusion.py                          # comp + ssl + edge + cell
python train_fusion.py --features comp          # single-modality baseline

# 5. Evaluate (7-class / 3-class)
python evaluate.py --split test --ckpt output_fusion/model_best.pth
python evaluate.py --split test --ckpt output_fusion/model_best.pth --three-class
```

## Citation

```bibtex
@article{liu2025osht,
  title   = {OSHT-Net: An OverLoCK-SCSA Backbone with a Hierarchical TransMIL Fusion
             Network for Breast Cancer Histopathological Image Classification},
  author  = {Liu, Hao and Li, Zihan and Yan, Bo and Zhang, Yuchen},
  journal = {...},
  year    = {2025}
}
```

## Acknowledgement

This code builds on [OverLoCK](https://github.com/LMMMEng/OverLoCK),
[timm](https://github.com/rwightman/pytorch-image-models),
[NATTEN](https://github.com/SHI-Labs/NATTEN),
[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet),
[Cellpose](https://github.com/MouseLand/cellpose), and
[UNI](https://github.com/mahmoodlab/UNI).

## License

See [LICENSE.md](LICENSE.md).
