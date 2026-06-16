import torch
import torch.nn as nn


class ChannelEncoderCNN(nn.Module):
    def __init__(self, emb_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.head(self.net(x))


class EEGFeatureFusionCNN(nn.Module):
    def __init__(
        self,
        eeg_channels: int = 21,
        emb_dim: int = 64,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.encoder = ChannelEncoderCNN(emb_dim=emb_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(eeg_channels * emb_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feats = []
        for ch in range(self.eeg_channels):
            x_ch = x[:, ch, :].unsqueeze(1)
            feats.append(self.encoder(x_ch))
        fused = torch.cat(feats, dim=1)
        return self.classifier(fused).squeeze(1)


if __name__ == "__main__":
    from torchsummary import summary

    model = EEGFeatureFusionCNN()
    summary(model, input_size=(21, 128))
