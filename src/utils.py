# (数据加载与工具)

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_pittsburgh_data(file_path):
    """
    加载 Excel 数据并进行预处理、划分和缩放。
    """
    data = pd.read_excel(file_path)
    X = data[['Longitude', 'Latitude']].values
    y = data['Altitude'].values.reshape(-1, 1)

    # 记录边界，用于后续寻优范围限制
    bounds = {
        'lon': (X[:, 0].min(), X[:, 0].max()),
        'lat': (X[:, 1].min(), X[:, 1].max())
    }

    # 划分数据集
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化缩放器
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 拟合并转换
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # 转换为 PyTorch Tensor
    data_tensors = {
        'X_train': torch.FloatTensor(X_train_scaled),
        'X_test': torch.FloatTensor(X_test_scaled),
        'y_train': torch.FloatTensor(y_train_scaled),
        'y_test': torch.FloatTensor(y_test_scaled)
    }

    return data_tensors, scaler_X, scaler_y, bounds

def create_prediction_grid(bounds, res=100):
    """
    创建一个均匀网格用于结果可视化。
    """
    lon_grid = np.linspace(bounds['lon'][0], bounds['lon'][1], res)
    lat_grid = np.linspace(bounds['lat'][0], bounds['lat'][1], res)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    grid_points = np.c_[lon_mesh.ravel(), lat_mesh.ravel()]
    return grid_points, lon_mesh, lat_mesh