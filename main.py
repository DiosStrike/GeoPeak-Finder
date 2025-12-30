# 把这些模块串联起来：加载数据 -> 训练模型 -> 保存模型 -> 寻找最高点 -> 打印结果

import torch
import os
import numpy as np
from src import engine
from src import visualizer
from src.model import ElevationNet
from src.utils import load_pittsburgh_data, create_prediction_grid

def main():
    # --- 1. 初始化 ---
    DATA_PATH = "data/PittsburghMap.xlsx"
    MODEL_SAVE_PATH = "models/elevation_model.pth"
    os.makedirs("results", exist_ok=True)
    
    # --- 2. 数据与模型准备 ---
    print("🚀 Step 1: Loading Data & Model...")
    data_tensors, scaler_X, scaler_y, bounds = load_pittsburgh_data(DATA_PATH)
    model = ElevationNet()
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("🔥 Training model...")
        model, _ = engine.train_elevation_model(model, data_tensors['X_train'], data_tensors['y_train'])
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # --- 3. 核心计算 (AI 引擎) ---
    print("🏔️ Calculating: Highest Peak (Gradient Ascent)...")
    peak_loc, peak_alt = engine.find_highest_peak(model, scaler_X, scaler_y, bounds)
    
    print("🌊 Calculating: Flood Sinks (Gradient Descent)...")
    flood_sinks_scaled = engine.find_flooding_sinks(model, num_droplets=100)
    flood_sinks = scaler_X.inverse_transform(flood_sinks_scaled)

    print("⛰️ Calculating: Slope Magnitude (Autograd)...")
    grid_points, lon_mesh, lat_mesh = create_prediction_grid(bounds)
    grid_points_scaled = scaler_X.transform(grid_points)
    slope_values = engine.calculate_slope(model, grid_points_scaled)
    slope_mesh = slope_values.reshape(100, 100)

    # 准备地形高度数据
    grid_tensor = torch.FloatTensor(grid_points_scaled)
    model.eval()
    with torch.no_grad():
        alt_pred = scaler_y.inverse_transform(model(grid_tensor).numpy())
    altitude_mesh = alt_pred.reshape(100, 100)

    # --- 4. 生成三张独立报告图 ---
    print("🎨 Step 3: Generating Three Separate Engineering Maps...")

    # 【图 1】 基础梯度图：展示地形 + 梯度上升找到的最高点
    visualizer.plot_pittsburgh_contour(
        lon_mesh, lat_mesh, altitude_mesh, 
        peak_loc, peak_alt, 
        save_path="results/1_basic_gradient_map.png"
    )

    # 【图 2】 洪涝风险图：展示地形 + 蓝色汇水点分析
    visualizer.plot_flood_risk(
        lon_mesh, lat_mesh, altitude_mesh, 
        peak_loc, flood_sinks, 
        save_path="results/2_flood_risk_analysis.png"
    )

    # 【图 3】 滑坡风险图：展示地形 + 红色高危坡度警戒线
    visualizer.plot_landslide_risk(
        lon_mesh, lat_mesh, altitude_mesh, slope_mesh, 
        peak_loc, 
        save_path="results/3_landslide_risk_analysis.png"
    )

    print("-" * 50)
    print("✅ 所有分析完成！请在 results 文件夹查看三张专业图纸：")
    print("📂 1. 基础梯度图 (Gradient Map)")
    print("📂 2. 洪涝分析图 (Flood Risk)")
    print("📂 3. 滑坡预警图 (Landslide Risk)")
    print("-" * 50)

if __name__ == "__main__":
    main()