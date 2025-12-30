"""
GeoPeak-Finder: Main execution pipeline for terrain digital twin analysis.
Includes neural surface fitting, gradient-based optimization, and risk assessment.
"""

import torch
import os
import numpy as np
from src import engine
from src import visualizer
from src.model import ElevationNet
from src.utils import load_pittsburgh_data, create_prediction_grid

def main():
    # Configuration and directory initialization
    DATA_PATH = "data/PittsburghMap.xlsx"
    MODEL_SAVE_PATH = "models/elevation_model.pth"
    os.makedirs("results", exist_ok=True)
    
    # Data acquisition and model instantiation
    print("Step 1: Loading geospatial datasets and neural architecture...")
    data_tensors, scaler_X, scaler_y, bounds = load_pittsburgh_data(DATA_PATH)
    model = ElevationNet()
    
    # Conditional model training or state-dict loading
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("Executing model training pipeline...")
        model, _ = engine.train_elevation_model(model, data_tensors['X_train'], data_tensors['y_train'])
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Topographical analysis and optimization
    print("Step 2: Executing gradient-based optimization and slope analysis...")
    
    # Peak identification via Multi-start Gradient Ascent
    peak_loc, peak_alt = engine.find_highest_peak(model, scaler_X, scaler_y, bounds)
    
    # Sink detection for flood susceptibility analysis
    flood_sinks_scaled = engine.find_flooding_sinks(model, num_droplets=100)
    flood_sinks = scaler_X.inverse_transform(flood_sinks_scaled)

    # Slope magnitude computation via Autograd
    grid_points, lon_mesh, lat_mesh = create_prediction_grid(bounds)
    grid_points_scaled = scaler_X.transform(grid_points)
    slope_values = engine.calculate_slope(model, grid_points_scaled)
    slope_mesh = slope_values.reshape(100, 100)

    # Neural surface inference for visualization
    grid_tensor = torch.FloatTensor(grid_points_scaled)
    model.eval()
    with torch.no_grad():
        alt_pred = scaler_y.inverse_transform(model(grid_tensor).numpy())
    altitude_mesh = alt_pred.reshape(100, 100)

    # Engineering map generation
    print("Step 3: Exporting analytical visualization assets...")

    # Map 1: Topographical Gradient & Peak Identification
    visualizer.plot_pittsburgh_contour(
        lon_mesh, lat_mesh, altitude_mesh, 
        peak_loc, peak_alt, 
        save_path="results/1_basic_gradient_map.png"
    )

    # Map 2: Flood Susceptibility & Sink Analysis
    visualizer.plot_flood_risk(
        lon_mesh, lat_mesh, altitude_mesh, 
        peak_loc, flood_sinks, 
        save_path="results/2_flood_risk_analysis.png"
    )

    # Map 3: Landslide Susceptibility & Slope Criticality
    visualizer.plot_landslide_risk(
        lon_mesh, lat_mesh, altitude_mesh, slope_mesh, 
        peak_loc, 
        save_path="results/3_landslide_risk_analysis.png"
    )

    # Status summary
    print("Execution complete. Analytical assets generated in /results directory:")
    print("1. 1_basic_gradient_map.png")
    print("2. 2_flood_risk_analysis.png")
    print("3. 3_landslide_risk_analysis.png")

if __name__ == "__main__":
    main()