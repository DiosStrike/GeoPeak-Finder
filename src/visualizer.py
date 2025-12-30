# 可视化工具

import matplotlib.pyplot as plt
import numpy as np

def plot_pittsburgh_contour(lon_mesh, lat_mesh, altitude_mesh, peak_location, peak_altitude, save_path=None):
    """
    绘制匹兹堡地形等高线图并标注最高点。
    """
    plt.figure(figsize=(14, 10))

    # 1. 绘制填充等高线
    contour_filled = plt.contourf(lon_mesh, lat_mesh, altitude_mesh, 
                                 levels=25, cmap='jet')
    plt.colorbar(contour_filled, label='Altitude (m)')

    # 2. 绘制等高线线条
    contour_lines = plt.contour(lon_mesh, lat_mesh, altitude_mesh, 
                                levels=15, colors='black', 
                                alpha=0.8, linewidths=1) # 修正：生产环境线宽通常设为1-2更清晰
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')

    # 3. 标注最高点
    plt.plot(peak_location[0], peak_location[1], 
             marker='^', color='white', markersize=15, 
             markeredgecolor='black', markeredgewidth=2,
             label=f'Highest Peak ({peak_altitude:.1f}m)', zorder=5)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Pittsburgh Terrain Digital Twin (Neural Network Predicted)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_flood_risk(grid_x, grid_y, grid_z, peak_loc, sinks, save_path):
    """专门绘制洪涝风险图"""
    plt.figure(figsize=(12, 9))
    # 基础地形
    plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='terrain', alpha=0.8)
    plt.colorbar(label='Altitude (m)')
    
    # 标注最高点 (红星)
    plt.scatter(peak_loc[0], peak_loc[1], color='red', marker='*', s=200, label='Highest Peak', edgecolors='white')
    
    # 标注洪涝点 (蓝色点)
    plt.scatter(sinks[:, 0], sinks[:, 1], color='blue', s=50, alpha=0.8, label='Flood Sinks (Accumulation)')
    
    plt.title("Pittsburgh Flood Risk Analysis (Sinks Detection)", fontsize=15)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_landslide_risk(grid_x, grid_y, grid_z, slope_mesh, peak_loc, save_path):
    """专门绘制滑坡风险图：使用红色线条圈出高危坡度区"""
    plt.figure(figsize=(12, 9))
    # 1. 基础地形（调淡一点，突出红线）
    plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='terrain', alpha=0.5)
    
    # 2. 计算高风险阈值（例如：全城坡度前 15% 的区域定义为高风险）
    risk_threshold = np.percentile(slope_mesh, 85) 
    
    # 3. 用红色粗线画出风险边界
    # levels=[risk_threshold] 意味着只画出超过这个数值的边界线
    landslide_contour = plt.contour(grid_x, grid_y, slope_mesh, levels=[risk_threshold], 
                                    colors='red', linewidths=2)
    plt.clabel(landslide_contour, inline=True, fontsize=10, fmt="High Risk")
    
    # 4. 填充一个半透明的淡红色，增强视觉感
    plt.contourf(grid_x, grid_y, slope_mesh, levels=[risk_threshold, np.max(slope_mesh)], 
                 colors=['red'], alpha=0.2)

    plt.scatter(peak_loc[0], peak_loc[1], color='black', marker='*', s=150, label='Highest Peak')
    
    plt.title("Pittsburgh Landslide Risk Analysis (Steep Slope Detection)", fontsize=15)
    plt.legend(['Landslide Risk Zone (Steep)'])
    plt.savefig(save_path, dpi=300)
    plt.close()