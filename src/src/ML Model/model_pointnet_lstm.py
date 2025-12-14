import torch
import torch.nn as nn
import torch.nn.functional as F

class FramePointNet(nn.Module):
    def __init__(self, in_features, emb_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        """
        x shape: (batch, T, N, features)
        """
        B, T, N, F = x.shape
        
        # Flatten objects
        x = x.view(B*T*N, F)
        h = self.mlp(x)
        h = h.view(B, T, N, -1)
        
        # Max pool over objects → frame embedding
        frame_emb = h.max(dim=2).values   # (B, T, emb_dim)
        return frame_emb


class PointNetLSTMClassifier(nn.Module):
    def __init__(self, in_features, emb_dim=128, lstm_hidden=128, num_layers=1):
        super().__init__()
        self.pointnet = FramePointNet(in_features, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x: (batch, T, N, features)
        """
        frame_embeddings = self.pointnet(x)  # (B, T, emb_dim)

        lstm_out, _ = self.lstm(frame_embeddings)  
        rep_emb = lstm_out[:, -1, :]  # last frame output (B, 2*lstm_hidden)

        logits = self.classifier(rep_emb)
        return logits
