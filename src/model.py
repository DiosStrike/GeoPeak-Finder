"""
Neural network architecture definition for GeoPeak-Finder.
Implements a Multilayer Perceptron (MLP) for topographical surface regression.
"""

import torch.nn as nn

class ElevationNet(nn.Module):
    """
    Geospatial regressor designed to map coordinate vectors (Longitude, Latitude) 
    to scalar elevation values for digital twin surface representation.
    """
    def __init__(self, n_in: int = 2, n_out: int = 1):
        super(ElevationNet, self).__init__()
        
        # Sequential architecture with triple-layer hidden units
        # Configured with 64 neurons per layer and ReLU non-linear activation
        self.seq = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_out)
        )
    
    def forward(self, x):
        """
        Execute forward pass for surface elevation inference.
        
        Args:
            x: Input tensor containing normalized geospatial coordinates.
        Returns:
            Predicted elevation scalar value.
        """
        return self.seq(x)