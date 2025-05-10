import torch
import torch.nn as nn

class PoseCNN(nn.Module):
    """
    PoseCNN model for 6D pose estimation.
    Takes RGB+mask input and predicts translation (3D) and rotation (quaternion).
    """
    def __init__(self):
        super(PoseCNN, self).__init__()
        # Feature extractor (4 channels for RGB+mask)
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 320x240
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 160x120
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 80x60
        )
        
        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 480, 640)
            flatten_size = self.features(dummy).view(1, -1).shape[1]
        
        # Output heads
        self.fc_trans = nn.Linear(flatten_size, 3)  # Translation (x,y,z)
        self.fc_rot = nn.Linear(flatten_size, 4)    # Rotation (quaternion)

    def forward(self, x):
        """Forward pass through network"""
        # Extract features
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Predict translation and rotation
        trans = self.fc_trans(x)
        rot = self.fc_rot(x)
        
        # Normalize quaternion to unit norm
        rot = rot / torch.norm(rot, dim=1, keepdim=True)
        
        return trans, rot