import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, D_in, D_H1, D_H2, D_out, seed=0):
        """Initialize parameters and build model."""
        super().__init__()
        
        # Define random seed for reproducibility
        torch.manual_seed(seed)
        
        self.non_linearity = nn.ReLU
#        self.non_linearity = nn.PReLU

        self.layer1 = nn.Sequential(
            nn.Linear(D_in, D_H1),
            nn.BatchNorm1d(D_H1),
            self.non_linearity()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(D_H1, D_H2),
            nn.BatchNorm1d(D_H2),
            self.non_linearity()
        )

        self.layer3 = nn.Linear(D_H2, D_out)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Build a network that maps input -> output."""
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x