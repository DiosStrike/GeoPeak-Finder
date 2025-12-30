# (训练与寻优逻辑)

import torch
import torch.nn as nn
import torch.optim as optim

def train_elevation_model(model, x_train, y_train, epochs=2000, lr=0.001):
    """
    模型训练核心引擎。
    """
    criterion = nn.MSELoss() # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam 优化器
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        # 1. 前向传播
        pred = model(x_train)
        loss = criterion(pred, y_train)
        
        # 2. 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 400 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    return model, train_losses





import torch.optim as optim
import numpy as np

def find_highest_peak(model, scaler_X, scaler_y, bounds, 
                      num_restarts=30, num_iterations=1000, lr=0.01):
    """
    使用随机梯度上升法寻找模型预测的海拔最高点。
    """
    best_altitude = float('-inf')
    best_location = None
    
    model.eval()
    
    # 将缩放器的均值和标准差转为 Tensor 提高计算效率
    X_mean = torch.tensor(scaler_X.mean_, dtype=torch.float32)
    X_std = torch.tensor(scaler_X.scale_, dtype=torch.float32)

    for restart in range(num_restarts):
        # 1. 随机初始化位置
        lon_init = np.random.uniform(bounds['lon'][0], bounds['lon'][1])
        lat_init = np.random.uniform(bounds['lat'][0], bounds['lat'][1])
        
        location = torch.tensor([[lon_init, lat_init]], requires_grad=True, dtype=torch.float32)
        optimizer = optim.Adam([location], lr=lr)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            # 手动缩放输入坐标 (location_scaled)
            location_scaled = (location - X_mean) / X_std
            
            # 预测缩放后的海拔
            altitude_scaled = model(location_scaled)
            
            # 损失函数为海拔的负值（因为 optimizer 默认是最小化，我们要最大化海拔）
            loss = -altitude_scaled
            loss.backward()
            optimizer.step()

            # 约束坐标在采样范围内 (Clamping)
            with torch.no_grad():
                location[0, 0].clamp_(bounds['lon'][0], bounds['lon'][1])
                location[0, 1].clamp_(bounds['lat'][0], bounds['lat'][1])

        # 检查当前结果
        with torch.no_grad():
            final_loc_scaled = (location - X_mean) / X_std
            final_alt_scaled = model(final_loc_scaled)
            final_alt = scaler_y.inverse_transform(final_alt_scaled.numpy())[0, 0]

        if final_alt > best_altitude:
            best_altitude = final_alt
            best_location = location.detach().numpy()[0]
            
    return best_location, best_altitude


import torch

def find_flooding_sinks(model, num_droplets=50, iterations=100, learning_rate=0.01):
    """
    通过梯度下降寻找地形的局部最低点（汇水点）
    """
    model.eval()
    # 1. 随机在地图上洒下“水滴”作为起始点
    # 假设经纬度范围已归一化在 [0, 1] 之间
    droplets = torch.rand((num_droplets, 2), requires_grad=True)
    
    optimizer = torch.optim.SGD([droplets], lr=learning_rate)

    for i in range(iterations):
        optimizer.zero_grad()
        # 我们要最小化海拔高度，所以直接把预测值作为 Loss
        altitude = model(droplets)
        loss = altitude.sum() 
        
        loss.backward()
        optimizer.step()
        
        # 约束水滴不超出地图边界
        with torch.no_grad():
            droplets.clamp_(0, 1)

    # 返回最终汇集的坐标点
    return droplets.detach().numpy()


def calculate_slope(model, grid_points_scaled):
    """
    利用 PyTorch 自动微分计算全域坡度
    """
    model.eval()
    # 转换为 Tensor 并开启梯度追踪
    inputs = torch.FloatTensor(grid_points_scaled).requires_grad_(True)
    
    # 前向传播得到预测海拔
    altitudes = model(inputs)
    
    # 利用自动微分计算梯度: dz/dx 和 dz/dy
    gradients = torch.autograd.grad(
        outputs=altitudes, 
        inputs=inputs,
        grad_outputs=torch.ones_like(altitudes),
        create_graph=False
    )[0]
    
    # 计算梯度模长（即坡度）
    slope_mag = torch.sqrt(torch.sum(gradients**2, dim=1))
    
    return slope_mag.detach().numpy()