"""
Utility module for geospatial data processing and grid generation.
Implements automated preprocessing, feature scaling, and meshgrid construction pipelines.
"""

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_pittsburgh_data(file_path):
    """
    Load, preprocess, and normalize geospatial elevation datasets.
    
    Args:
        file_path: Path to the source Excel file containing geospatial coordinates.
    Returns:
        data_tensors: Dictionary containing train/test tensors.
        scaler_X: Fitted coordinate transformer.
        scaler_y: Fitted elevation transformer.
        bounds: Geospatial boundary constraints for optimization.
    """
    # Dataset acquisition
    data = pd.read_excel(file_path)
    X = data[['Longitude', 'Latitude']].values
    y = data['Altitude'].values.reshape(-1, 1)

    # Determine geospatial boundary constraints for gradient-based optimization
    bounds = {
        'lon': (X[:, 0].min(), X[:, 0].max()),
        'lat': (X[:, 1].min(), X[:, 1].max())
    }

    # Train-test split implementation
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling initialization (Z-score normalization)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fitting and transformation of spatial features and targets
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # Tensor conversion for PyTorch computational graph integration
    data_tensors = {
        'X_train': torch.FloatTensor(X_train_scaled),
        'X_test': torch.FloatTensor(X_test_scaled),
        'y_train': torch.FloatTensor(y_train_scaled),
        'y_test': torch.FloatTensor(y_test_scaled)
    }

    return data_tensors, scaler_X, scaler_y, bounds

def create_prediction_grid(bounds, res=100):
    """
    Generate a uniform meshgrid for spatial inference and topographical visualization.
    
    Args:
        bounds: Dictionary containing geospatial coordinate limits.
        res: Grid resolution for interpolation.
    """
    # Linear spacing across geospatial dimensions
    lon_grid = np.linspace(bounds['lon'][0], bounds['lon'][1], res)
    lat_grid = np.linspace(bounds['lat'][0], bounds['lat'][1], res)
    
    # Coordinate matrix construction
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    grid_points = np.c_[lon_mesh.ravel(), lat_mesh.ravel()]
    
    return grid_points, lon_mesh, lat_mesh