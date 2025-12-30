"""
Visualizer module for geospatial risk assessment and topographical mapping.
Implements standardized plotting routines for peak identification, hydraulic sinks, 
and slope criticality analysis.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_pittsburgh_contour(lon_mesh, lat_mesh, altitude_mesh, peak_location, peak_altitude, save_path=None):
    """
    Generate topographical contour maps with peak identification for geospatial analysis.
    
    Args:
        lon_mesh: Longitude coordinate matrix.
        lat_mesh: Latitude coordinate matrix.
        altitude_mesh: Predicted elevation surface matrix.
        peak_location: Coordinates of the global maximum.
        peak_altitude: Scalar value of the maximum elevation.
        save_path: Directory path for exporting the generated figure.
    """
    plt.figure(figsize=(14, 10))

    # 1. Filled contour representation for elevation gradient
    contour_filled = plt.contourf(lon_mesh, lat_mesh, altitude_mesh, 
                                 levels=25, cmap='jet')
    plt.colorbar(contour_filled, label='Altitude (m)')

    # 2. Primary contour isolines with interval labeling
    contour_lines = plt.contour(lon_mesh, lat_mesh, altitude_mesh, 
                                levels=15, colors='black', 
                                alpha=0.8, linewidths=1)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')

    # 3. Marker for global elevation maximum identified via optimization
    plt.plot(peak_location[0], peak_location[1], 
             marker='^', color='white', markersize=15, 
             markeredgecolor='black', markeredgewidth=2,
             label=f'Identified Peak ({peak_altitude:.1f}m)', zorder=5)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Topographical Digital Twin: Pittsburgh Regional Elevation Surface', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Asset exported to: {save_path}")
    
    plt.show()

def plot_flood_risk(grid_x, grid_y, grid_z, peak_loc, sinks, save_path):
    """
    Visualize flood susceptibility based on hydraulic sink accumulation points.
    
    Args:
        grid_x: Longitude meshgrid.
        grid_y: Latitude meshgrid.
        grid_z: Neural surface elevation mesh.
        peak_loc: Reference peak coordinates.
        sinks: Identified topographical minima (accumulation points).
    """
    plt.figure(figsize=(12, 9))
    
    # Topographical base layer for flood modeling
    plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='terrain', alpha=0.8)
    plt.colorbar(label='Altitude (m)')
    
    # Reference peak marker
    plt.scatter(peak_loc[0], peak_loc[1], color='red', marker='*', s=200, label='Elevation Peak', edgecolors='white')
    
    # Hydraulic sink mapping representing localized accumulation risk
    plt.scatter(sinks[:, 0], sinks[:, 1], color='blue', s=50, alpha=0.8, label='Hydraulic Sinks (Flood Susceptibility)')
    
    plt.title("Hydraulic Analysis: Flood Accumulation Sink Identification", fontsize=15)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_landslide_risk(grid_x, grid_y, grid_z, slope_mesh, peak_loc, save_path):
    """
    Visualize landslide susceptibility through critical slope gradient detection.
    
    Args:
        grid_x: Longitude meshgrid.
        grid_y: Latitude meshgrid.
        grid_z: Neural surface elevation mesh.
        slope_mesh: Computed gradient magnitude matrix.
    """
    plt.figure(figsize=(12, 9))
    
    # Topographical base layer with reduced opacity for risk highlighting
    plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='terrain', alpha=0.5)
    
    # Determination of critical slope threshold (85th percentile distribution)
    risk_threshold = np.percentile(slope_mesh, 85) 
    
    # Delineation of high-risk landslide zones (Critical slope isolines)
    landslide_contour = plt.contour(grid_x, grid_y, slope_mesh, levels=[risk_threshold], 
                                    colors='red', linewidths=2)
    plt.clabel(landslide_contour, inline=True, fontsize=10, fmt="CRITICAL SLOPE")
    
    # Translucent risk zone fill for enhanced spatial detection
    plt.contourf(grid_x, grid_y, slope_mesh, levels=[risk_threshold, np.max(slope_mesh)], 
                 colors=['red'], alpha=0.2)

    plt.scatter(peak_loc[0], peak_loc[1], color='black', marker='*', s=150, label='Reference Peak')
    
    plt.title("Geotechnical Analysis: Landslide Susceptibility & Slope Criticality", fontsize=15)
    plt.legend(['Critical Slope Threshold (>85th Percentile)'])
    plt.savefig(save_path, dpi=300)
    plt.close()