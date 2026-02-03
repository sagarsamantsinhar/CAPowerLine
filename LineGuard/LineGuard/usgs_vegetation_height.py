"""
Get vegetation height from USGS 3DEP Lidar data.

This module provides functions to query vegetation height at specific locations
using USGS 3D Elevation Program (3DEP) lidar data.

USGS 3DEP provides:
- Digital Elevation Model (DEM) - bare earth elevation
- Digital Surface Model (DSM) - top of canopy/structures elevation
- Vegetation height = DSM - DEM

Requires: pip install rasterio requests --break-system-packages
"""

import requests
import json
from typing import Tuple, Optional, Dict
import os


class VegetationHeightAPI:
    """
    Access vegetation height using USGS 3DEP web services.
    """
    
    # USGS 3DEP Elevation Point Query Service
    USGS_ELEVATION_API = "https://epqs.nationalmap.gov/v1/json"
    
    # USGS TNM (The National Map) API for lidar data
    TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_elevation_usgs(self, lat: float, lon: float, dataset: str = "ned") -> Optional[float]:
        """
        Get elevation at a point using USGS Elevation Point Query Service.
        
        Args:
            lat: Latitude
            lon: Longitude  
            dataset: Dataset to use ('ned', '3dep', etc.)
            
        Returns:
            Elevation in meters, or None if unavailable
        """
        params = {
            'x': lon,
            'y': lat,
            'units': 'Meters',
            'output': 'json'
        }
        
        try:
            response = self.session.get(self.USGS_ELEVATION_API, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'value' in data:
                elevation = data['value']
                if elevation != -1000000:  # USGS returns this for no data
                    return float(elevation)
        except Exception as e:
            print(f"Error querying USGS elevation API: {e}")
        
        return None
    
    def find_available_lidar(self, lat: float, lon: float, 
                            radius_km: float = 5) -> list:
        """
        Find available lidar datasets near a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            
        Returns:
            List of available lidar products
        """
        # Convert km to degrees (approximate)
        radius_deg = radius_km / 111.0
        
        bbox = f"{lon-radius_deg},{lat-radius_deg},{lon+radius_deg},{lat+radius_deg}"
        
        params = {
            'bbox': bbox,
            'prodFormats': 'LAS,LAZ',  # Lidar formats
            'datasets': '3D Elevation Program (3DEP) - Lidar Point Cloud',
            'outputFormat': 'JSON',
            'max': 10
        }
        
        try:
            response = self.session.get(self.TNM_API, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            items = data.get('items', [])
            return items
        except Exception as e:
            print(f"Error finding lidar data: {e}")
            return []


def get_vegetation_height_from_files(lat: float, lon: float, 
                                     dem_file: str, 
                                     dsm_file: str) -> Optional[float]:
    """
    Calculate vegetation height from local DEM and DSM raster files.
    
    This requires you to have downloaded the lidar-derived rasters:
    - DEM (Digital Elevation Model) - bare earth
    - DSM (Digital Surface Model) - top of surface/canopy
    
    Args:
        lat: Latitude
        lon: Longitude
        dem_file: Path to DEM GeoTIFF file
        dsm_file: Path to DSM GeoTIFF file
        
    Returns:
        Vegetation height in meters, or None if unavailable
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        print("Error: rasterio not installed. Run: pip install rasterio --break-system-packages")
        return None
    
    try:
        # Read DEM (bare earth elevation)
        with rasterio.open(dem_file) as dem:
            row, col = rowcol(dem.transform, lon, lat)
            dem_value = dem.read(1)[row, col]
        
        # Read DSM (surface elevation)
        with rasterio.open(dsm_file) as dsm:
            row, col = rowcol(dsm.transform, lon, lat)
            dsm_value = dsm.read(1)[row, col]
        
        # Check for no-data values
        if dem_value == dem.nodata or dsm_value == dsm.nodata:
            return None
        
        # Vegetation height = DSM - DEM
        veg_height = dsm_value - dem_value
        
        # Sanity check (vegetation height should be positive and reasonable)
        if veg_height < 0:
            veg_height = 0
        
        return float(veg_height)
        
    except Exception as e:
        print(f"Error reading raster files: {e}")
        return None


def get_vegetation_height_from_point_cloud(lat: float, lon: float,
                                          las_file: str,
                                          search_radius: float = 5.0) -> Optional[Dict]:
    """
    Calculate vegetation height from LAS/LAZ point cloud file.
    
    This reads the raw lidar point cloud and calculates:
    - Ground elevation (classification = 2)
    - Vegetation top (classification = 3, 4, 5)
    - Vegetation height
    
    Args:
        lat: Latitude
        lon: Longitude
        las_file: Path to LAS or LAZ file
        search_radius: Search radius in meters
        
    Returns:
        Dictionary with ground_elevation, canopy_elevation, and vegetation_height
        or None if unavailable
    """
    try:
        import laspy
        import numpy as np
        from pyproj import Transformer
    except ImportError:
        print("Error: laspy not installed. Run: pip install laspy[laszip] --break-system-packages")
        return None
    
    try:
        # Read LAS file
        las = laspy.read(las_file)
        
        # Get coordinate system info
        # Most USGS lidar is in UTM or State Plane, need to transform lat/lon
        if hasattr(las, 'header') and hasattr(las.header, 'parse_crs'):
            crs = las.header.parse_crs()
            
            # Transform lat/lon to file CRS
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            target_x, target_y = transformer.transform(lon, lat)
        else:
            # Assume coordinates are already in same system (less accurate)
            target_x, target_y = lon, lat
        
        # Get points within search radius
        x = las.x
        y = las.y
        z = las.z
        classification = las.classification
        
        # Calculate distance from target point
        distances = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        nearby_mask = distances <= search_radius
        
        if not np.any(nearby_mask):
            print(f"No points found within {search_radius}m of location")
            return None
        
        # Get ground points (classification = 2)
        ground_mask = nearby_mask & (classification == 2)
        if np.any(ground_mask):
            ground_elevation = np.mean(z[ground_mask])
        else:
            # Use lowest point as ground estimate
            ground_elevation = np.min(z[nearby_mask])
        
        # Get vegetation points (classification 3=low veg, 4=medium veg, 5=high veg)
        veg_mask = nearby_mask & np.isin(classification, [3, 4, 5])
        
        if np.any(veg_mask):
            canopy_elevation = np.max(z[veg_mask])
            vegetation_height = canopy_elevation - ground_elevation
        else:
            # No vegetation classified, use all points above ground
            canopy_elevation = np.max(z[nearby_mask])
            vegetation_height = canopy_elevation - ground_elevation
        
        return {
            'ground_elevation': float(ground_elevation),
            'canopy_elevation': float(canopy_elevation),
            'vegetation_height': float(max(0, vegetation_height)),
            'points_analyzed': int(np.sum(nearby_mask)),
            'ground_points': int(np.sum(ground_mask)),
            'vegetation_points': int(np.sum(veg_mask))
        }
        
    except Exception as e:
        print(f"Error processing point cloud: {e}")
        return None


def download_lidar_for_location(lat: float, lon: float, 
                                output_dir: str = "./lidar_data") -> Dict[str, str]:
    """
    Find and download lidar data for a specific location.
    
    NOTE: This function helps you find download URLs. Actual downloads may be large (GB).
    
    Args:
        lat: Latitude
        lon: Longitude
        output_dir: Directory to save information
        
    Returns:
        Dictionary with download information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    api = VegetationHeightAPI()
    
    print(f"Searching for lidar data near ({lat}, {lon})...")
    products = api.find_available_lidar(lat, lon)
    
    if not products:
        print("No lidar data found for this location.")
        print("\nAlternative options:")
        print("1. Check USGS EarthExplorer: https://earthexplorer.usgs.gov/")
        print("2. Check OpenTopography: https://opentopography.org/")
        print("3. Check state/local lidar repositories")
        return {}
    
    print(f"\nFound {len(products)} lidar product(s):")
    
    result = {
        'products': [],
        'download_urls': []
    }
    
    for i, product in enumerate(products):
        title = product.get('title', 'Unknown')
        download_url = product.get('downloadURL', '')
        size_mb = product.get('sizeInBytes', 0) / (1024 * 1024)
        
        print(f"\n{i+1}. {title}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   URL: {download_url}")
        
        result['products'].append({
            'title': title,
            'url': download_url,
            'size_mb': size_mb
        })
        result['download_urls'].append(download_url)
    
    # Save to file
    info_file = os.path.join(output_dir, f'lidar_info_{lat}_{lon}.json')
    with open(info_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Saved download info to: {info_file}")
    print("\nTo download, use wget or curl with the URLs above.")
    
    return result


# Simple API-based function (limited accuracy, no vegetation-specific data)
def get_terrain_elevation(lat: float, lon: float) -> Optional[float]:
    """
    Simple function to get terrain elevation using USGS API.
    
    NOTE: This returns ground elevation only, NOT vegetation height.
    For vegetation height, you need DSM - DEM from lidar data.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Ground elevation in meters, or None if unavailable
    """
    api = VegetationHeightAPI()
    return api.get_elevation_usgs(lat, lon)


def main():
    """Example usage"""
    # Fort Dick, CA coordinates
    lat = 41.868074
    lon = -124.152736
    
    print("=" * 70)
    print("USGS LIDAR VEGETATION HEIGHT LOOKUP")
    print("=" * 70)
    print()
    
    # Method 1: Get ground elevation (API-based, always available)
    print("Method 1: Getting ground elevation from USGS API...")
    elevation = get_terrain_elevation(lat, lon)
    if elevation:
        print(f"✓ Ground elevation: {elevation:.2f} meters")
    else:
        print("✗ Could not retrieve elevation")
    print()
    
    # Method 2: Find available lidar data
    print("Method 2: Searching for lidar data...")
    info = download_lidar_for_location(lat, lon)
    print()
    
    # Method 3: If you have local files (example)
    print("Method 3: Using local lidar files (if available)...")
    dem_file = "path/to/dem.tif"
    dsm_file = "path/to/dsm.tif"
    
    if os.path.exists(dem_file) and os.path.exists(dsm_file):
        veg_height = get_vegetation_height_from_files(lat, lon, dem_file, dsm_file)
        if veg_height is not None:
            print(f"✓ Vegetation height: {veg_height:.2f} meters")
    else:
        print("  (Local DEM/DSM files not found - this is expected)")
        print("  To use this method:")
        print("  1. Download lidar products from URLs above")
        print("  2. Extract DEM and DSM rasters")
        print("  3. Update file paths in the code")
    print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
For vegetation height, you need:
1. Download lidar data for your area (use download_lidar_for_location)
2. Process to get DEM (bare earth) and DSM (surface)
3. Use get_vegetation_height_from_files(lat, lon, dem, dsm)

Alternatively:
- Use get_vegetation_height_from_point_cloud() for raw LAS/LAZ files
- This gives you direct access to point classifications
    """)


if __name__ == '__main__':
    main()