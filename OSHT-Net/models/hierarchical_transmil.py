"""Hierarchical TransMIL fusion network (Sec. 3.4 of the paper).

SingleStreamTransMIL aggregates one variable-length feature sequence into a
[CLS] token; HierarchicalTransMIL first aggregates each modality independently
and then fuses the modality tokens with a global fusion Transformer.
"""
import torch
import torch.nn as nn


class SingleStreamTransMIL(nn.Module):
    """One single-stream Transformer that maps a sequence (N, d_in) to one token."""

    def __init__(self, input_dim: int, hidden_dim: int = 8, dropout: float = 0.2,
                 nhead: int = 4, num_layers: int = 1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        out = self.transformer(x)
        return out[:, 0, :]


class HierarchicalTransMIL(nn.Module):
    """Dual-level TransMIL: intra-modality aggregation + inter-modality fusion."""

    def __init__(self, feature_dims: dict, hidden_dim: int = 8, dropout: float = 0.2,
                 num_classes: int = 7, nhead: int = 4, fusion_layers: int = 2):
        """
        Args:
            feature_dims: dict mapping modality name -> input feature dim,
                e.g. {'comp': 1024, 'ssl': 1024, 'edge': 384, 'cell': 1024}.
        """
        super().__init__()
        self.feature_names = list(feature_dims.keys())
        self.streams = nn.ModuleDict({
            name: SingleStreamTransMIL(dim, hidden_dim, dropout, nhead)
            for name, dim in feature_dims.items()
        })
        self.global_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True)
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat_dict: dict) -> torch.Tensor:
        """feat_dict: modality name -> Tensor (B, N_k, d_k)."""
        device = next(self.parameters()).device
        stream_outputs = []
        for name in self.feature_names:
            x = feat_dict[name].to(device)
            cls_out = self.streams[name](x)
            stream_outputs.append(cls_out.unsqueeze(1))

        fused_seq = torch.cat(stream_outputs, dim=1)                      # (B, K, hidden)
        global_cls = self.global_cls_token.expand(fused_seq.shape[0], -1, -1)
        fused_seq = torch.cat((global_cls, fused_seq), dim=1)             # (B, K+1, hidden)
        fusion_out = self.fusion_transformer(fused_seq)
        logits = self.classifier(fusion_out[:, 0, :])
        return logits
