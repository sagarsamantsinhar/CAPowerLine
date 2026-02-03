"""
Get current vegetation height from ICESat-2 satellite data.

ICESat-2 provides much more current data than USGS lidar:
- USGS lidar: 5-8 years old (last flight 2018-2019 for most areas)
- ICESat-2: Updated every 91 days (data from last week available!)

Trade-off: ICESat-2 only provides measurements along narrow satellite tracks,
not wall-to-wall coverage like airborne lidar.

Requires: pip install sliderule pandas geopandas --break-system-packages
"""

import pandas as pd
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta


def get_icesat2_vegetation_sliderule(lat: float, 
                                     lon: float,
                                     buffer_km: float = 2.0,
                                     date_start: Optional[str] = None,
                                     date_end: Optional[str] = None) -> pd.DataFrame:
    """
    Get ICESat-2 vegetation height data using SlideRule (easiest method).
    
    Args:
        lat: Latitude
        lon: Longitude
        buffer_km: Search radius in kilometers (default 2km)
        date_start: Start date 'YYYY-MM-DD' (default: 1 year ago)
        date_end: End date 'YYYY-MM-DD' (default: today)
    
    Returns:
        DataFrame with vegetation measurements
    """
    try:
        from sliderule import icesat2, gedi
    except ImportError:
        print("Error: sliderule not installed")
        print("Install with: pip install sliderule --break-system-packages")
        return pd.DataFrame()
    
    # Initialize SlideRule
    icesat2.init("slideruleearth.io", verbose=False)
    
    # Default date range: last year
    if date_end is None:
        date_end = datetime.now().strftime('%Y-%m-%d')
    if date_start is None:
        date_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Fetching ICESat-2 data for ({lat}, {lon})")
    print(f"Search radius: {buffer_km} km")
    print(f"Date range: {date_start} to {date_end}")
    
    # Convert buffer to degrees (approximate)
    buffer_deg = buffer_km / 111.0
    
    # Define region (simple box around point)
    region = [
        {"lon": lon - buffer_deg, "lat": lat - buffer_deg},
        {"lon": lon + buffer_deg, "lat": lat - buffer_deg},
        {"lon": lon + buffer_deg, "lat": lat + buffer_deg},
        {"lon": lon - buffer_deg, "lat": lat + buffer_deg},
        {"lon": lon - buffer_deg, "lat": lat - buffer_deg},  # Close polygon
    ]
    
    # Request parameters
    parms = {
        "poly": region,
        "t0": date_start,
        "t1": date_end,
        "cnf": 4,  # Surface confidence (4 = high quality)
        "ats": 10.0,  # Along-track spread
        "cnt": 5,  # Minimum photon count
        "len": 40.0,  # Segment length (meters)
        "res": 20.0,  # Resolution (meters)
    }
    
    try:
        # Get ATL08 data (land and vegetation height)
        print("Requesting data from NASA...")
        atl08 = icesat2.atl08p(parms)
        
        if atl08.empty:
            print("No ICESat-2 data found for this location and time period")
            print("\nPossible reasons:")
            print("  - Location not yet covered by ICESat-2 tracks")
            print("  - Cloud cover during satellite passes")
            print("  - Try increasing buffer_km or date range")
            return pd.DataFrame()
        
        print(f"✓ Found {len(atl08)} measurements")
        
        # Calculate vegetation height
        atl08['vegetation_height_m'] = atl08['h_canopy'] - atl08['h_te_mean']
        
        # Filter out negative/unrealistic values
        atl08 = atl08[atl08['vegetation_height_m'] >= 0]
        atl08 = atl08[atl08['vegetation_height_m'] <= 100]  # Max 100m trees
        
        # Add distance from query point
        atl08['distance_km'] = (
            ((atl08['latitude'] - lat) ** 2 + 
             (atl08['longitude'] - lon) ** 2) ** 0.5 * 111
        )
        
        # Sort by distance
        atl08 = atl08.sort_values('distance_km')
        
        return atl08
        
    except Exception as e:
        print(f"Error fetching ICESat-2 data: {e}")
        return pd.DataFrame()


def get_nearest_vegetation_height(lat: float, lon: float, 
                                  max_distance_km: float = 5.0) -> Optional[Dict]:
    """
    Get the nearest ICESat-2 vegetation height measurement.
    
    Args:
        lat: Latitude
        lon: Longitude
        max_distance_km: Maximum search distance (km)
    
    Returns:
        Dictionary with nearest measurement or None
    """
    df = get_icesat2_vegetation_sliderule(lat, lon, buffer_km=max_distance_km)
    
    if df.empty:
        return None
    
    # Get nearest measurement
    nearest = df.iloc[0]
    
    result = {
        'latitude': float(nearest['latitude']),
        'longitude': float(nearest['longitude']),
        'vegetation_height_m': float(nearest['vegetation_height_m']),
        'canopy_height_m': float(nearest['h_canopy']),
        'ground_elevation_m': float(nearest['h_te_mean']),
        'distance_km': float(nearest['distance_km']),
        'date': str(nearest.get('time', 'Unknown')),
        'quality': 'high' if nearest.get('quality_flag', 0) == 0 else 'medium',
        'data_source': 'ICESat-2 ATL08'
    }
    
    return result


def get_vegetation_statistics(lat: float, lon: float,
                              buffer_km: float = 2.0) -> Optional[Dict]:
    """
    Get vegetation height statistics from multiple ICESat-2 measurements.
    
    Args:
        lat: Latitude
        lon: Longitude
        buffer_km: Search radius in kilometers
    
    Returns:
        Dictionary with statistics or None
    """
    df = get_icesat2_vegetation_sliderule(lat, lon, buffer_km=buffer_km)
    
    if df.empty:
        return None
    
    veg_heights = df['vegetation_height_m']
    
    stats = {
        'mean_height_m': float(veg_heights.mean()),
        'median_height_m': float(veg_heights.median()),
        'min_height_m': float(veg_heights.min()),
        'max_height_m': float(veg_heights.max()),
        'std_dev_m': float(veg_heights.std()),
        'num_measurements': len(veg_heights),
        'search_radius_km': buffer_km,
        'center_lat': lat,
        'center_lon': lon,
        'data_source': 'ICESat-2 ATL08'
    }
    
    return stats


def compare_icesat2_with_lidar(lat: float, lon: float,
                               dem_file: str,
                               dsm_file: str) -> Dict:
    """
    Compare ICESat-2 (current) with USGS lidar (historical).
    
    Args:
        lat: Latitude
        lon: Longitude
        dem_file: Path to USGS DEM file (2018-2019)
        dsm_file: Path to USGS DSM file (2018-2019)
    
    Returns:
        Dictionary with comparison
    """
    # Get historical lidar data
    try:
        from usgs_vegetation_height import get_vegetation_height_from_files
        lidar_height = get_vegetation_height_from_files(lat, lon, dem_file, dsm_file)
        lidar_year = 2018  # Typical for Northern CA
    except Exception as e:
        print(f"Could not get lidar data: {e}")
        lidar_height = None
        lidar_year = None
    
    # Get current ICESat-2 data
    icesat2_result = get_nearest_vegetation_height(lat, lon)
    
    if icesat2_result and lidar_height:
        change = icesat2_result['vegetation_height_m'] - lidar_height
        years = 2024 - lidar_year
        annual_change = change / years
        
        return {
            'lidar_height_m': lidar_height,
            'lidar_year': lidar_year,
            'icesat2_height_m': icesat2_result['vegetation_height_m'],
            'icesat2_year': 2024,
            'change_m': change,
            'annual_change_m_per_year': annual_change,
            'years_elapsed': years,
            'icesat2_distance_km': icesat2_result['distance_km']
        }
    
    return {
        'lidar_height_m': lidar_height,
        'icesat2_result': icesat2_result,
        'comparison': 'Could not compare - missing data'
    }


def main():
    """Example usage for Fort Dick, CA"""
    
    print("=" * 70)
    print("ICESat-2 VEGETATION HEIGHT QUERY")
    print("=" * 70)
    print()
    
    # Fort Dick, CA coordinates
    lat = 41.868074
    lon = -124.152736
    
    print(f"Location: Fort Dick, CA")
    print(f"Coordinates: {lat}, {lon}")
    print()
    
    # Method 1: Get nearest measurement
    print("Method 1: Nearest ICESat-2 Measurement")
    print("-" * 70)
    result = get_nearest_vegetation_height(lat, lon, max_distance_km=5.0)
    
    if result:
        print(f"✓ Found measurement:")
        print(f"  Vegetation height: {result['vegetation_height_m']:.2f} meters")
        print(f"  Ground elevation: {result['ground_elevation_m']:.2f} meters")
        print(f"  Canopy elevation: {result['canopy_height_m']:.2f} meters")
        print(f"  Distance from query: {result['distance_km']:.2f} km")
        print(f"  Date: {result['date']}")
        print(f"  Quality: {result['quality']}")
    else:
        print("✗ No ICESat-2 data found within 5 km")
        print("  Try increasing max_distance_km or check different time period")
    
    print()
    
    # Method 2: Get statistics from multiple measurements
    print("Method 2: Statistics from Multiple Measurements")
    print("-" * 70)
    stats = get_vegetation_statistics(lat, lon, buffer_km=3.0)
    
    if stats:
        print(f"✓ Found {stats['num_measurements']} measurements within {stats['search_radius_km']} km:")
        print(f"  Mean height: {stats['mean_height_m']:.2f} ± {stats['std_dev_m']:.2f} meters")
        print(f"  Median height: {stats['median_height_m']:.2f} meters")
        print(f"  Range: {stats['min_height_m']:.2f} - {stats['max_height_m']:.2f} meters")
    else:
        print("✗ No ICESat-2 data found")
    
    print()
    print("=" * 70)
    print("NOTES")
    print("=" * 70)
    print("""
ICESat-2 provides:
  ✓ Very current data (updated every 91 days)
  ✓ Good vertical accuracy (~3-5m for vegetation)
  ✓ Global coverage
  
Limitations:
  ✗ Sparse spatial coverage (only along satellite tracks)
  ✗ Not wall-to-wall like airborne lidar
  ✗ May have gaps due to clouds
  
Best for: Point measurements, time series, change detection
Not ideal for: Complete area mapping, high-resolution canopy maps
    """)


if __name__ == '__main__':
    main()
