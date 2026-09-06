import torch
import torch.nn as nn

class TacticalGestureMLP(nn.Module):
    def __init__(self, num_classes=4):
        super(TacticalGestureMLP, self).__init__()
        
        # Lightweight 3-Layer Dense Network (Runs in < 1ms on CPU / ESP32 ready)
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)