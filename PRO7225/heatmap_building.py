"""
PIR - Projet d'Initiation a la recherche @ Telecom Paris
Code 03 - Building heatmaps with information from height, azimuth, and system type

Author: Alvaro RIBAS
"""

# 0 - Imports =============================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from rasterio.transform import from_origin
from rasterio.features import rasterize
from owslib.wfs import WebFeatureService
from pyproj import Transformer
import geopandas as gpd
from PIL import Image
from scipy.constants import c
from scipy.special import i0 # Modified Bessel function Ia of order a=0
from scipy.stats import vonmises
import math

# 1 - Importing maps from server to make building height map ==============================================================

# 1.1 - Configuration

min_distance_map = pd.read_csv('min_distance_map.csv')
area_size_m = 500 # meters
img_size    = 256 # number of pixels
wfs_url     = "https://data.geopf.fr/wfs/ows"
layer_name  = "BDTOPO_V3:batiment"

# 1.2 - Preparing reprojection
transformer_to_m   = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
transformer_to_deg = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
wfs = WebFeatureService(url=wfs_url, version="2.0.0")

# 1.3 - Saving maps
os.makedirs("building_maps_test", exist_ok=True)

maps = []
for idx, (lat, lon) in enumerate(zip(min_distance_map['Latitude'], min_distance_map['Longitude'])):
    # 1.3.1 - Compute bounding box in lon/lat
    x_c, y_c = transformer_to_m.transform(lon, lat)
    half = area_size_m / 2
    lon_min, lat_min = transformer_to_deg.transform(x_c-half, y_c-half)
    lon_max, lat_max = transformer_to_deg.transform(x_c+half, y_c+half)
    bbox = (lon_min, lat_min, lon_max, lat_max)

    # 1.3.2 - pull down building footprints
    resp = wfs.getfeature(
        typename=layer_name,
        bbox=bbox,
        outputFormat="application/json"
    )
    gdf = gpd.read_file(resp)

    # 1.3.3 - estimate heights
    def estimate_height(r):
        if r.get('hauteur') is not None:
            return r['hauteur']
        if r.get('nombre_etages') is not None:
            return r['nombre_etages'] * 3
        return np.nan

    gdf['height'] = gdf.apply(estimate_height, axis=1)
    gdf['height'].fillna(gdf['height'].mean(), inplace=True)

    # 1.3.4 - rasterize to a (img_size, img_size) array
    px_w = (lon_max - lon_min) / img_size
    px_h = (lat_max - lat_min) / img_size
    transform = from_origin(lon_min, lat_max, px_w, px_h)
    shapes = [(geom, h) for geom, h in zip(gdf.geometry, gdf.height)]
    height_raster = rasterize(
        shapes,
        out_shape=(img_size, img_size),
        transform=transform,
        fill=0,
        dtype='float32'
    )

    # 1.3.5 - convert to torch tensor [1, H, W]
    t = torch.from_numpy(height_raster).unsqueeze(0)
    maps.append(t)

    # 1.3.6 -  save as PNG for inspection
    #    (normalize to 0–255 for uint8 image)
    arr = height_raster
    arr_n = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    img = Image.fromarray(arr_n)
    img.save(f"building_maps_test/building_map_{idx:04d}.png")

# 1.3.7 - stack into a single tensor of shape (N,1,256,256) for training
images = torch.stack(maps, dim=0)
print(f"Built {len(maps)} building‐height maps → images.shape = {images.shape}")

# 2 - Building the antenna heatmaps with information from height, azimuth, and system
# ==================================

# Loading datasets
min_distance_map = pd.read_csv('min_distance_map.csv')
df_sites_antennas = pd.read_csv('df_sites_antennas.csv')

# Image parameters
img_size = 256
WORLD_SIZE_METERS = 500  # Each image represents 400x400 meters

# Define range multipliers for different systems
# LTE systems (9.0, 10.0, 11.0) have base range
# 5G NR systems have 2x range
LTE_SYSTEMS = [9.0, 10.0, 11.0]
BASE_MAX_RADIUS = 512  # Maximum beam extent in pixels for LTE
NR_MAX_RADIUS = 512     # Maximum beam extent in pixels for 5G NR (2x)

# Intensity parameters
LTE_BASE_INTENSITY = 0.75  # LTE normalized intensity at antenna
NR_BASE_INTENSITY = 1.0   # 5G NR normalized intensity at antenna

# Beam parameters
kappa = 2  # Von Mises concentration
a = 64  # Cardioid parameter for LTE
a_nr = 128  # Cardioid parameter for 5G NR (2x)
phi_angles = [0, 2*np.pi/3, 4*np.pi/3]  # For cardioid: 0°, 120°, 240°

# Measurement points (longitude, latitude)
measurement_lon = np.array(min_distance_map['Longitude'])
measurement_lat = np.array(min_distance_map['Latitude'])

# Nearest antenna information for each measurement point
nearest_antenna_lon = np.array(min_distance_map['Nearest Longitude'])
nearest_antenna_lat = np.array(min_distance_map['Nearest Latitude'])

# Create a lookup dictionary for antenna properties
antenna_lookup = {}
for idx in range(len(df_sites_antennas)):
    key = (df_sites_antennas.loc[idx, 'Longitude'], df_sites_antennas.loc[idx, 'Latitude'])
    antenna_lookup[key] = {
        'system': df_sites_antennas.loc[idx, 'Système'],
        'azimuth': df_sites_antennas.loc[idx, 'Azimut'],
        'height': df_sites_antennas.loc[idx, 'Hauteur / sol']
    }

# Functions

def meters_per_degree_lat():
    """Meters per degree of latitude (constant everywhere)"""
    return 111320  # meters per degree latitude

def meters_per_degree_lon(lat):
    """Meters per degree of longitude at given latitude"""
    return 111320 * np.cos(np.radians(lat))

def lonlat_to_pixel(lon, lat, center_lon, center_lat, img_size, world_size_meters):
    """Convert lon/lat to pixel coordinates relative to center
    
    Args:
        lon, lat: coordinates to convert
        center_lon, center_lat: center point coordinates
        img_size: image size in pixels (256)
        world_size_meters: real-world size in meters (400)
    
    Returns:
        px, py: pixel coordinates
    """
    # Calculate offset in degrees
    dlon = lon - center_lon
    dlat = lat - center_lat
    
    # Convert to meters
    # For longitude: need to account for latitude
    meters_lon = dlon * meters_per_degree_lon(center_lat)
    meters_lat = dlat * meters_per_degree_lat()
    
    # Convert meters to pixels
    # world_size_meters corresponds to img_size pixels
    meters_per_pixel = world_size_meters / img_size
    
    px = img_size / 2 + meters_lon / meters_per_pixel
    py = img_size / 2 + meters_lat / meters_per_pixel
    
    return px, py

def is_in_image(px, py, img_size):
    """Check if pixel coordinates are within image bounds"""
    return 0 <= px < img_size and 0 <= py < img_size

def compute_radial_attenuation(r_pixels, base_intensity, img_size, world_size_meters, max_radius_pixels):
    """Compute radial attenuation based on 1/r^2 law
    
    Args:
        r_pixels: distance in pixels from antenna
        base_intensity: base intensity at antenna (0.5 for LTE, 1.0 for 5G NR)
        img_size: image size in pixels
        world_size_meters: real-world size in meters
        max_radius_pixels: maximum radius in pixels for this system
    
    Returns:
        attenuation factor based on distance
    """
    # Convert pixel distance to kilometers
    meters_per_pixel = world_size_meters / img_size
    r_meters = r_pixels * meters_per_pixel
    r_km = r_meters / 1000.0
    
    # Avoid division by zero at antenna location
    # Use a small minimum distance (e.g., 1 meter = 0.001 km)
    r_km = np.maximum(r_km, 0.001)
    
    # Apply 1/r^2 law with base intensity
    # At r=0.001km: intensity = base_intensity / (0.001^2) (very high, will be clipped by normalization)
    # At r=0.5km: intensity = base_intensity / (0.5^2) = base_intensity / 0.25 = 4 * base_intensity
    attenuation = base_intensity * (-20*np.log(r_km))
    
    # Apply hard cutoff at max_radius
    attenuation[r_pixels > max_radius_pixels] = 0
    
    return attenuation

def compute_cardioid_pattern(x_grid, y_grid, center_x, center_y, a, phi, max_radius, base_intensity, img_size, world_size_meters):
    """Compute cardioid radiation pattern with radial attenuation from antenna at (center_x, center_y)
    with main lobe pointing in direction phi"""
    x_centered = x_grid - center_x
    y_centered = y_grid - center_y
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)
    
    # Angular pattern: cardioid
    angular_intensity = a * (1 + np.cos(theta - phi))
    
    # Normalize angular pattern to [0, 1]
    if angular_intensity.max() > 0:
        angular_intensity = angular_intensity / angular_intensity.max()
    
    # Radial attenuation based on 1/r^2
    radial_attenuation = compute_radial_attenuation(r, base_intensity, img_size, world_size_meters, max_radius)
    
    # Combine angular and radial components
    intensity = angular_intensity * radial_attenuation
    
    return intensity

def compute_vonmises_pattern(x_grid, y_grid, center_x, center_y, mu, kappa, max_radius, base_intensity, img_size, world_size_meters):
    """Compute von Mises directional beam with radial attenuation from antenna at (center_x, center_y)
    pointing in direction mu"""
    x_centered = x_grid - center_x
    y_centered = y_grid - center_y
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)
    
    # Angular intensity using von Mises
    angular_intensity = vonmises.pdf(theta, kappa, loc=mu)
    # Normalize so peak is at 1
    if vonmises.pdf(0, kappa) > 0:
        angular_intensity = angular_intensity / vonmises.pdf(0, kappa)
    
    # Radial attenuation based on 1/r^2
    radial_attenuation = compute_radial_attenuation(r, base_intensity, img_size, world_size_meters, max_radius)
    
    # Combine angular and radial components
    intensity = angular_intensity * radial_attenuation
    
    return intensity

def find_closest_cardioid_angle(antenna_azimuth_rad, measurement_angle, phi_angles):
    """Find which of the three cardioid lobes is closest to the measurement point"""
    # Calculate the angle of each cardioid lobe
    lobe_angles = [(antenna_azimuth_rad + phi) % (2 * np.pi) for phi in phi_angles]
    
    # Find angular distance to each lobe
    angular_distances = []
    for lobe_angle in lobe_angles:
        # Calculate smallest angular distance (accounting for wraparound)
        diff = abs(measurement_angle - lobe_angle)
        diff = min(diff, 2*np.pi - diff)
        angular_distances.append(diff)
    
    # Return the phi_offset that gives the closest lobe
    closest_idx = np.argmin(angular_distances)
    return phi_angles[closest_idx]

# Process each measurement point
num_measurements = len(measurement_lon)
all_beam_patterns = []

print(f"Processing {num_measurements} measurement points...")

for idx in range(num_measurements):
    if idx % 100 == 0:
        print(f"Processing measurement {idx}/{num_measurements}")
    
    center_lon = measurement_lon[idx]
    center_lat = measurement_lat[idx]
    
    # Get nearest antenna for this measurement point
    nearest_lon = nearest_antenna_lon[idx]
    nearest_lat = nearest_antenna_lat[idx]
    
    # Look up antenna properties
    antenna_key = (nearest_lon, nearest_lat)
    
    # Skip if antenna not found in lookup
    if antenna_key not in antenna_lookup:
        print(f"Warning: Antenna at ({nearest_lon}, {nearest_lat}) not found for measurement {idx}")
        all_beam_patterns.append(np.zeros((img_size, img_size)))
        continue
    
    antenna_props = antenna_lookup[antenna_key]
    system = antenna_props['system']
    azimuth = antenna_props['azimuth']
    
    # Determine range and cardioid parameter based on system
    is_lte = system in LTE_SYSTEMS
    max_radius = BASE_MAX_RADIUS if is_lte else NR_MAX_RADIUS
    cardioid_a = a if is_lte else a_nr
    base_intensity = LTE_BASE_INTENSITY if is_lte else NR_BASE_INTENSITY
    
    # Create coordinate grids CENTERED ON MEASUREMENT POINT
    y, x = np.ogrid[:img_size, :img_size]
    
    # Initialize beam pattern
    beam_pattern = np.zeros((img_size, img_size))
    
    # Convert antenna position to pixel coordinates (relative to measurement point at center)
    ant_px, ant_py = lonlat_to_pixel(
        nearest_lon, nearest_lat,
        center_lon, center_lat, 
        img_size, WORLD_SIZE_METERS
    )
    
    # Generate beam pattern based on system type
    if is_lte:
        # For LTE: use cardioid pattern, but only the lobe closest to measurement point
        azimuth_rad = np.deg2rad(azimuth)
        
        # Calculate angle from antenna to measurement point
        dx = img_size / 2 - ant_px
        dy = img_size / 2 - ant_py
        measurement_angle = np.arctan2(dy, dx)
        
        # Find the closest cardioid lobe
        closest_phi = find_closest_cardioid_angle(azimuth_rad, measurement_angle, phi_angles)
        
        # Generate only the closest cardioid with radial attenuation
        beam_pattern = compute_cardioid_pattern(x, y, ant_px, ant_py, cardioid_a, 
                                                azimuth_rad + closest_phi, max_radius,
                                                base_intensity, img_size, WORLD_SIZE_METERS)
    else:
        # For 5G NR: use von Mises beam pointing toward measurement center
        dx = img_size / 2 - ant_px
        dy = img_size / 2 - ant_py
        mu_to_center = np.arctan2(dy, dx)
        beam_pattern = compute_vonmises_pattern(x, y, ant_px, ant_py, mu_to_center, kappa, max_radius,
                                               base_intensity, img_size, WORLD_SIZE_METERS)
    
    # Store the beam pattern (DO NOT normalize here - keep absolute intensities)
    all_beam_patterns.append(beam_pattern)

print("Saving beam patterns...")

# Save all beam patterns as individual PNG files
for idx, beam_pattern in enumerate(all_beam_patterns):
    if idx % 100 == 0:
        print(f"Saving beam pattern {idx}/{len(all_beam_patterns)}")
    plt.figure(figsize=(img_size/100, img_size/100), dpi=100)
    plt.imshow(beam_pattern, origin='lower', cmap='viridis', vmin=0, vmax=None)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'beam_pattern_{idx}.png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

# Plot the first measurement point as an example
if len(all_beam_patterns) > 0:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    fig, ax = plt.subplots(figsize=(10, 8))
    
    idx = 0  # First measurement point
    center_lon = measurement_lon[idx]
    center_lat = measurement_lat[idx]
    nearest_lon = nearest_antenna_lon[idx]
    nearest_lat = nearest_antenna_lat[idx]
    
    # Get antenna properties
    antenna_key = (nearest_lon, nearest_lat)
    if antenna_key in antenna_lookup:
        system = antenna_lookup[antenna_key]['system']
        system_type = "LTE" if system in LTE_SYSTEMS else "5G NR"
        base_int = LTE_BASE_INTENSITY if system in LTE_SYSTEMS else NR_BASE_INTENSITY
    else:
        system_type = "Unknown"
        base_int = 0
    
    # Plot beam pattern
    im = ax.imshow(all_beam_patterns[idx], origin='lower', extent=[0, img_size, 0, img_size], 
                   cmap='viridis', vmin=0)
    
    # Measurement point is at center
    ax.plot(img_size/2, img_size/2, 'r*', markersize=15, markeredgecolor='white', 
            markeredgewidth=2, label='Measurement Point (Center)')
    
    # Mark nearest antenna position
    ant_px, ant_py = lonlat_to_pixel(
        nearest_lon, nearest_lat,
        center_lon, center_lat,
        img_size, WORLD_SIZE_METERS
    )
    if is_in_image(ant_px, ant_py, img_size):
        ax.plot(ant_px, ant_py, 'o', markersize=10, color='red', 
                markeredgecolor='white', markeredgewidth=2, label='Nearest Antenna')
    
    ax.set_title(f'Beam Pattern - Measurement Point 1\n({center_lon:.4f}°, {center_lat:.4f}°) - {system_type}')
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized Signal Intensity')
    
    plt.tight_layout()
    plt.savefig('example_beam_pattern.png', dpi=150, bbox_inches='tight')
    plt.show()
    
print(f"\nComplete!")
print(f"Saved {len(all_beam_patterns)} beam pattern images as 'beam_pattern_*.png'")
print(f"Each image represents a 400m × 400m area at 256×256 pixels")
print(f"LTE base intensity: {LTE_BASE_INTENSITY}, 5G NR base intensity: {NR_BASE_INTENSITY}")
print(f"Radial decay: 1/r² (r in km, max range ~0.5 km)")
print(f"Saved example plot as 'example_beam_pattern.png'")