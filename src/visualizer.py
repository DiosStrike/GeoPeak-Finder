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