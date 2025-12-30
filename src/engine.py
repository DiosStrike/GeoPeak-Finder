"""
Core computational engine for topographical analysis.
Implements neural network training, gradient-based optimization, and spatial risk analytics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_elevation_model(model, x_train, y_train, epochs=2000, lr=0.001):
    """
    Execute neural network training pipeline for elevation surface fitting.
    
    Args:
        model: Neural network architecture for topographical representation.
        x_train: Input coordinate tensors (normalized).
        y_train: Target elevation tensors (normalized).
        epochs: Number of training iterations.
        lr: Learning rate for Adam optimizer.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        # Forward pass
        pred = model(x_train)
        loss = criterion(pred, y_train)
        
        # Backpropagation and weight optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 400 == 0:
            print(f"Iteration [{epoch+1}/{epochs}], MSE Loss: {loss.item():.4f}")
            
    return model, train_losses

def find_highest_peak(model, scaler_X, scaler_y, bounds, 
                      num_restarts=30, num_iterations=1000, lr=0.01):
    """
    Implement Multi-start Stochastic Gradient Ascent (SGA) to identify global maximum.
    
    Args:
        model: Trained neural elevation surface.
        scaler_X: Coordinate normalization transformer.
        scaler_y: Elevation normalization transformer.
        bounds: Geospatial boundary constraints.
    """
    best_altitude = float('-inf')
    best_location = None
    
    model.eval()
    
    # Pre-compute normalization statistics as tensors for computational efficiency
    X_mean = torch.tensor(scaler_X.mean_, dtype=torch.float32)
    X_std = torch.tensor(scaler_X.scale_, dtype=torch.float32)

    for restart in range(num_restarts):
        # Uniform random initialization within geospatial bounds
        lon_init = np.random.uniform(bounds['lon'][0], bounds['lon'][1])
        lat_init = np.random.uniform(bounds['lat'][0], bounds['lat'][1])
        
        location = torch.tensor([[lon_init, lat_init]], requires_grad=True, dtype=torch.float32)
        optimizer = optim.Adam([location], lr=lr)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            # Manual coordinate scaling for differentiable inference
            location_scaled = (location - X_mean) / X_std
            
            # Inference on neural surface
            altitude_scaled = model(location_scaled)
            
            # Minimization of negative altitude for gradient ascent
            loss = -altitude_scaled
            loss.backward()
            optimizer.step()

            # Boundary enforcement via coordinate clamping
            with torch.no_grad():
                location[0, 0].clamp_(bounds['lon'][0], bounds['lon'][1])
                location[0, 1].clamp_(bounds['lat'][0], bounds['lat'][1])

    # Result verification and inverse transformation
    with torch.no_grad():
        final_loc_scaled = (location - X_mean) / X_std
        final_alt_scaled = model(final_loc_scaled)
        final_alt = scaler_y.inverse_transform(final_alt_scaled.numpy())[0, 0]

    if final_alt > best_altitude:
        best_altitude = final_alt
        best_location = location.detach().numpy()[0]
            
    return best_location, best_altitude

def find_flooding_sinks(model, num_droplets=50, iterations=100, learning_rate=0.01):
    """
    Identify topographical minima (hydraulic sinks) via gradient descent optimization.
    
    Args:
        model: Trained neural elevation surface.
        num_droplets: Number of stochastic initialization points.
        iterations: Optimization steps for convergence.
    """
    model.eval()
    # Stochastic initialization of particle coordinates in normalized space [0, 1]
    droplets = torch.rand((num_droplets, 2), requires_grad=True)
    
    optimizer = torch.optim.SGD([droplets], lr=learning_rate)

    for i in range(iterations):
        optimizer.zero_grad()
        # Direct minimization of neural elevation output
        altitude = model(droplets)
        loss = altitude.sum() 
        
        loss.backward()
        optimizer.step()
        
        # Spatial constraint enforcement
        with torch.no_grad():
            droplets.clamp_(0, 1)

    return droplets.detach().numpy()

def calculate_slope(model, grid_points_scaled):
    """
    Compute topographical slope magnitude using automatic differentiation.
    
    Args:
        model: Trained neural elevation surface.
        grid_points_scaled: Input coordinate grid for slope evaluation.
    """
    model.eval()
    # Tensor conversion and gradient tracking initialization
    inputs = torch.FloatTensor(grid_points_scaled).requires_grad_(True)
    
    # Forward pass to retrieve surface elevation
    altitudes = model(inputs)
    
    # Partial derivative computation via autograd: dz/dx and dz/dy
    gradients = torch.autograd.grad(
        outputs=altitudes, 
        inputs=inputs,
        grad_outputs=torch.ones_like(altitudes),
        create_graph=False
    )[0]
    
    # Vector magnitude calculation for slope intensity
    slope_mag = torch.sqrt(torch.sum(gradients**2, dim=1))
    
    return slope_mag.detach().numpy()