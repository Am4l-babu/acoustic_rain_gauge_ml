"""
CNN / LSTM / Transformer regression heads for MFCC input
============================================================
Reimplemented cleanly from SARID's paper description (Section 3.2), not
imported from their code: external_repos/SARID/baseline.py and
utils/dataloader.py contain unfilled template placeholders
(f'{data features train}.npy' isn't valid Python) and reference undefined
classes (USRADataset_CR), so they don't run as-is. Their nets/general_net.py
CNN block and dataloader.py's LSTM/Transformer also have dead code and shape
bugs (e.g. an AdaptiveAvgPool2d applied to a 3D tensor).

All three models take MFCC input of shape (B, n_mfcc, T_frames) -- for our
data, (B, 40, 157) -- and output a single scalar rainfall_mm estimate.
"""

import torch
import torch.nn as nn


class CNNRegressor(nn.Module):
    """3 conv layers + BN/ReLU, pooling+dropout, GAP, 2 FC layers -- matches
    the paper's described CNN structure (Section 3.2.1)."""

    def __init__(self, n_mfcc: int = 40):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                 # (B, 1, n_mfcc, T)
        x = self.features(x).flatten(1)    # (B, 128)
        return self.fc(x).squeeze(-1)


class LSTMRegressor(nn.Module):
    """2 stacked LSTM layers, 256 hidden units -- matches Section 3.2.2.
    Regresses off the final layer's last hidden state (standard
    sequence-to-one pattern)."""

    def __init__(self, n_mfcc: int = 40, hidden_size: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_mfcc, hidden_size=hidden_size,
                             num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)              # (B, T, n_mfcc)
        _out, (h_n, _c_n) = self.lstm(x)
        last_hidden = h_n[-1]              # (B, hidden_size)
        return self.fc(last_hidden).squeeze(-1)


class TransformerRegressor(nn.Module):
    """4-layer Transformer encoder, nhead=4, dim_feedforward=512 -- matches
    Section 3.2.3. Global-average-pools the encoder output over time before
    the final FC layer."""

    def __init__(self, n_mfcc: int = 40, nhead: int = 4, num_layers: int = 4,
                 dim_feedforward: int = 512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_mfcc, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(n_mfcc, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)              # (B, T, n_mfcc)
        out = self.encoder(x)              # (B, T, n_mfcc)
        pooled = out.mean(dim=1)           # (B, n_mfcc)
        return self.fc(pooled).squeeze(-1)


MODEL_REGISTRY = {
    "cnn": CNNRegressor,
    "lstm": LSTMRegressor,
    "transformer": TransformerRegressor,
}
