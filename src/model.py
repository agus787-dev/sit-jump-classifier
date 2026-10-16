"""Model definition.

Idea: run the SAME small CNN on every frame, average the frame features over
time, then classify the averaged feature vector.

    input  [B, T, 3, H, W]
      -> FrameEncoder on each frame   [B*T, 32]
      -> reshape                      [B, T, 32]
      -> mean over T                  [B, 32]
      -> classifier                   [B, num_classes]
"""

import torch
import torch.nn as nn

FEATURE_DIM = 32   # size of the feature vector produced for each frame


class FrameEncoder(nn.Module):
    """A tiny CNN that turns one frame into a feature vector."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input frame shape: [N, 3, 128, 128]
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Shape: [N, 16, 64, 64]
            nn.Conv2d(16, FEATURE_DIM, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Shape: [N, 32, 32, 32]
            nn.AdaptiveAvgPool2d((1, 1)),
            # Shape: [N, 32, 1, 1]
        )

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, start_dim=1)   # [N, 32]


class SimpleVideoCNN(nn.Module):
    """Frame encoder + temporal average pooling + small classifier."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.frame_encoder = FrameEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape

        # Merge batch and time so the CNN sees every frame as a separate image.
        frames = x.reshape(batch_size * time_steps, channels, height, width)
        frame_features = self.frame_encoder(frames)

        # Split them back: [B*T, F] -> [B, T, F]
        frame_features = frame_features.reshape(batch_size, time_steps, -1)

        # Average over time: [B, T, F] -> [B, F]
        video_features = frame_features.mean(dim=1)

        return self.classifier(video_features)   # logits [B, num_classes]


def print_model_summary(model: nn.Module):
    """Print the structure and the number of parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel structure:")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
