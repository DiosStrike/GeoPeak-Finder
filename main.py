# 把这些模块串联起来：加载数据 -> 训练模型 -> 保存模型 -> 寻找最高点 -> 打印结果

import torch
import os
from src.model import ElevationNet
from src.engine import train_elevation_model, find_highest_peak
from src.utils import load_pittsburgh_data, create_prediction_grid
from src.visualizer import plot_pittsburgh_contour

def main():
    # --- 1. 环境准备 ---
    DATA_PATH = "data/PittsburghMap.xlsx"
    MODEL_SAVE_PATH = "models/elevation_model.pth"
    
    # --- 2. 加载与预处理数据 ---
    print("Loading and preprocessing data...")
    data_tensors, scaler_X, scaler_y, bounds = load_pittsburgh_data(DATA_PATH)
    
    # --- 3. 初始化模型与训练 ---
    print("Initializing Neural Network...")
    model = ElevationNet()
    
    # 如果已有训练好的模型则加载，否则训练
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("Starting training...")
        model, _ = train_elevation_model(
            model, 
            data_tensors['X_train'], 
            data_tensors['y_train']
        )
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- 4. 寻找最高点 (寻优逻辑) ---
    print("Executing Gradient Ascent to find peak...")
    peak_loc, peak_alt = find_highest_peak(
        model, scaler_X, scaler_y, bounds
    )
    
    # --- 5. 结果展示 ---
    print("-" * 30)
    print(f"Peak Found at: Lon {peak_loc[0]:.4f}, Lat {peak_loc[1]:.4f}")
    print(f"Predicted Altitude: {peak_alt:.2f} m")
    print(f"Location Name: Robert Williams Reservoir")
    print(f"Google Maps: https://www.google.com/maps?q={peak_loc[1]},{peak_loc[0]}")
    print("-" * 30)

    # --- 6. 可视化 ---
    print("Generating contour map...")
    grid_points, lon_mesh, lat_mesh = create_prediction_grid(bounds)
    
    # 预测网格高度
    grid_tensor = torch.FloatTensor(scaler_X.transform(grid_points))
    model.eval()
    with torch.no_grad():
        alt_pred_scaled = model(grid_tensor)
        alt_pred = scaler_y.inverse_transform(alt_pred_scaled.numpy())
    
    altitude_mesh = alt_pred.reshape(100, 100)
    
    plot_pittsburgh_contour(
        lon_mesh, lat_mesh, altitude_mesh, 
        peak_loc, peak_alt, 
        save_path="models/contour_result.png"
    )

if __name__ == "__main__":
    main()