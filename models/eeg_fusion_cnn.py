import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self, in_features, out_features: int, dropout: float = 0.2):
        super(Classifier, self).__init__()
        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.tail(x)


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding:int = 1):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.block(x)


class CNNBlocks(nn.Module):
    def __init__(self, in_channels: int):
        super(CNNBlocks, self).__init__()
        self.net = nn.Sequential(
            CNNBlock(in_channels, 16),
            CNNBlock(16, 32),
            CNNBlock(32, 64)
        )

    def forward(self, x):
        return self.net(x)


class EEGChannelInputFusion(nn.Module):
    def __init__(self, eeg_channels: int = 21, out_features: int = 1, dropout: float = 0.2):
        super(EEGChannelInputFusion, self).__init__()
        self.net = nn.Sequential(
            # Fusion Layer
            nn.Conv1d(eeg_channels, 1, kernel_size=1),
            # CNN Blocks
            CNNBlocks(1),
            # Classifier
            Classifier(64, out_features, dropout=dropout),
        )

    def forward(self, x):
        return self.net(x)


class EEGChannelFeatureFusion(nn.Module):
    def __init__(self, eeg_channels: int = 21, out_features: int = 1, dropout: float = 0.2):
        super(EEGChannelFeatureFusion, self).__init__()
        self.net = nn.Sequential(
            # CNN Blocks
            CNNBlocks(eeg_channels),
            # Fusion Layer
            nn.Conv1d(64, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.LeakyReLU(),
            # Classifier
            Classifier(64, out_features, dropout=dropout),
        )

    def forward(self, x):
        return self.net(x)

class BaselineCNN(nn.Module):
    """
    1D-CNN Backbone for extracting spatial and local morphological features 
    from multi-channel EEG windows.
    """
    def __init__(self, num_channels: int = 21, time_steps: int = 128, dropout: float = 0.3):
        super(BaselineCNN, self).__init__()
        # Input Fusion: Project 21 channels down to 16
        self.data_fusion = nn.Conv1d(num_channels, 16, kernel_size=1)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1), 
            nn.BatchNorm1d(32),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2) 
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm1d(64),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2)  
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2)  
        )
        self.flatten = nn.Flatten()
        
        # Dynamically calculate the flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, time_steps)
            dummy = self.conv3(self.conv2(self.conv1(self.data_fusion(dummy))))
            self.feature_size = dummy.view(1, -1).size(1)
            
        self.fc1 = nn.Sequential(
            nn.Linear(self.feature_size, 128), 
            nn.ReLU(),
            nn.Dropout(p=dropout)  
        ) 

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 128-dimensional feature vector for a given EEG window."""
        x = self.data_fusion(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return self.fc1(x)
        


class EEG_CNN_LSTM(nn.Module):
    """
    Spatiotemporal Deep Learning Architecture for Seizure Detection.
    Uses a Time-Distributed CNN backbone followed by a Bidirectional LSTM.
    """
    def __init__(self, eeg_channels: int = 21, out_features: int = 1, dropout: float = 0.5):
        super(EEG_CNN_LSTM, self).__init__()
        
        cnn_dropout = dropout * 0.6 
        self.backbone = BaselineCNN(num_channels=eeg_channels, time_steps=128, dropout=cnn_dropout)
        
        hidden_size = 64
        self.lstm = nn.LSTM(
            input_size=128,          
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True       
        )
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional means the output dimension is hidden_size * 2
        lstm_out_dim = hidden_size * 2
        self.classifier = nn.Linear(lstm_out_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the 4D input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Sequence, Channels, Time)
                              e.g., [B, 5, 21, 128].
                              
        Returns:
            torch.Tensor: Logits of shape (Batch, out_features).
        """
        # Ensure the input is at least 4D [B, S, C, T]
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        B, S, C, T = x.size()
        
        # 1. Time-Distributed CNN execution via tensor flattening
        x_flat = x.view(B * S, C, T)
        features = self.backbone.extract_features(x_flat)
        
        # 2. Reshape back to sequence format for LSTM
        features = features.view(B, S, -1)
        
        # 3. Temporal modeling
        lstm_out, _ = self.lstm(features)
        
        # 4. Average pooling across the sequence dimension for stable prediction
        mean_out = torch.mean(lstm_out, dim=1) 
        mean_out = self.dropout(mean_out)
        
        # Returns raw logits (to be consumed by BCEWithLogitsLoss)
        return self.classifier(mean_out)
    
    
if __name__ == "__main__":
    from torchsummary import summary

    model = EEGChannelInputFusion()
    summary(model, input_size=(21, 128))