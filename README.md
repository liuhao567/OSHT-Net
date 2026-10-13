# OSHT-Net: An OverLoCK-SCSA Backbone with a Hierarchical TransMIL Fusion Network for Breast Cancer Histopathological Image Classification




OSHT-Net classifies a breast histopathology ROI into seven subtypes (N, PB, UDH, FEA, ADH, DCIS, IC). One ROI is mapped to four complementary feature sequences: **comp** (tissue texture, OverLoCK-SCSA supervised on BRACS), **ssl** (self-supervised semantics, MAE-pretrained), **edge** (boundary structure, ViT-Small), and **cell** (nuclear morphology, Cellpose + UNI). **Hierarchical TransMIL** first aggregates each sequence into a modality token, then fuses the four tokens for the final seven-class prediction. The backbone follows the “overview-first, look-closely-next” design, enhanced with **SCSA** spatial-channel attention and a **Context Bridge-SCSA** module at the entry of each Focus-Net sub-stage.

**Main results (BRACS validation):** 7-class Balanced Accuracy **65.51**, Weighted F1 **66.10**, Macro Precision **67.86**, Macro AUC **91.90**; 3-class **86.32 / 85.63 / 85.95 / 94.99**. The Bridge-SCSA backbone alone reaches **70.60 / 69.78 / 70.61 / 92.98**. Fine-tuning on BACH reaches **83.3%** accuracy.

**Repository contents:** the full training and evaluation pipeline — backbone training, MAE pretraining, four-branch feature extraction, Hierarchical TransMIL fusion training, and 7-class / 3-class evaluation. Dataset preparation, pretrained weights, and step-by-step reproduction follow the paper in `README.md`.



