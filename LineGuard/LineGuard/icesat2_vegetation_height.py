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
    Get ICESat-2 vegetation height data using SlideRule.
    
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
        from sliderule import icesat2
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
        {"lon": lon - buffer_deg, "lat": lat - buffer_deg},
    ]
    
    # Request parameters for ATL08
    parms = {
        "poly": region,
        "t0": date_start,
        "t1": date_end,
        "cnf": 4,  # Surface confidence
        "ats": 10.0,
        "cnt": 5,
        "len": 40.0,
        "res": 20.0,
    }
    
    try:
        print("Requesting ATL08 data from NASA...")
        atl08 = icesat2.atl08p(parms)
        
        if atl08.empty:
            print("No ICESat-2 data found for this location and time period")
            #print("\nPossible reasons:")
            #print("  - Location not yet covered by ICESat-2 tracks")
            #print("  - Cloud cover during satellite passes")
            #print("  - Try increasing buffer_km or expanding date range")
            return pd.DataFrame()
        
        print(f"✓ Found {len(atl08)} measurements")
        
        # Print available columns for debugging
        #print(f"\nAvailable columns: {list(atl08.columns)[:200]}...")
        
        # SlideRule ATL08 field names (these are the actual field names)
        # Based on SlideRule documentation and ICESat-2 ATL08 specification
        
        # Try different possible field name combinations
        terrain_field = None
        canopy_field = None
        
        # Check for terrain elevation field
        # Preference order: median > mean > best_fit (median is more robust to outliers)
        for field in ['h_te_median', 'h_te_mean', 'terrain_h', 'h_te_best_fit', 'dem_h']:
            if field in atl08.columns:
                terrain_field = field
                break
        
        # Check for canopy field
        for field in ['h_canopy', 'canopy_h', 'h_max_canopy', 'h_mean_canopy']:
            if field in atl08.columns:
                canopy_field = field
                break
        
        if terrain_field is None or canopy_field is None:
            #print("\n❌ Error: Could not find expected field names!")
            #print(f"All available columns:")
            for i, col in enumerate(atl08.columns):
                print(f"  {i+1}. {col}")
            print("\nPlease report these column names so the script can be updated.")
            
            # Try to save a sample for inspection
            try:
                atl08.head(5).to_csv('icesat2_sample_output.csv', index=False)
                print("\n✓ Saved sample to 'icesat2_sample_output.csv' for inspection")
            except:
                pass
            
            return pd.DataFrame()
        
        #print(f"Using terrain field: '{terrain_field}'")
        #print(f"Using canopy field: '{canopy_field}'")
        
        # Calculate vegetation height
        atl08['vegetation_height_m'] = atl08[canopy_field] - atl08[terrain_field]
        
        atl08['ground_elevation_m'] = atl08[terrain_field]
        atl08['canopy_elevation_m'] = atl08[canopy_field]
        
        # Filter out negative/unrealistic values
        before_filter = len(atl08)
        atl08 = atl08[atl08['vegetation_height_m'] >= 1]
        atl08 = atl08[atl08['vegetation_height_m'] <= 100]  # Max 100m trees
        after_filter = len(atl08)
        
        #if before_filter > after_filter:
        #    print(f"Filtered out {before_filter - after_filter} unrealistic values")

        if after_filter> 0:
            print(f"Found {after_filter} realistic values")
        
        # Add distance from query point
        #atl08['distance_km'] = (
        #    ((atl08['latitude'] - lat) ** 2 + 
        #     (atl08['longitude'] - lon) ** 2) ** 0.5 * 111
        #)
        
        # Sort by distance
        #atl08 = atl08.sort_values('distance_km')
        
        return atl08
        
    except Exception as e:
        print(f"Error fetching ICESat-2 data: {e}")
        import traceback
        traceback.print_exc()
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
        'vegetation_height_m': float(nearest['vegetation_height_m']),
        'canopy_elevation_m': float(nearest['canopy_elevation_m']),
        'ground_elevation_m': float(nearest['ground_elevation_m']),
        #'distance_km': float(nearest['distance_km']),
        'date': str(nearest.get('time', 'Unknown')),
        'quality': 'high',
        'data_source': 'ICESat-2 ATL08 via SlideRule'
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
        'data_source': 'ICESat-2 ATL08 via SlideRule'
    }
    
    return stats


def save_results_to_csv(lat: float, lon: float, 
                        buffer_km: float = 5.0,
                        output_file: str = "icesat2_vegetation.csv"):
    """
    Get ICESat-2 data and save to CSV file.
    
    Args:
        lat: Latitude
        lon: Longitude
        buffer_km: Search radius in km
        output_file: Output CSV filename
    """
    df = get_icesat2_vegetation_sliderule(lat, lon, buffer_km=buffer_km)
    
    if df.empty:
        print("No data to save")
        return
    
    # Select relevant columns
    columns_to_save = [
        'latitude', 'longitude', 
        'vegetation_height_m', 'canopy_elevation_m', 'ground_elevation_m',
        'distance_km', 'time'
    ]
    
    # Only include columns that exist
    columns_to_save = [col for col in columns_to_save if col in df.columns]
    
    df[columns_to_save].to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(df)} measurements to {output_file}")


def main():
    """Example usage for Fort Dick, CA"""
    
    print("=" * 70)
    print("ICESat-2 VEGETATION HEIGHT QUERY - CORRECTED VERSION")
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
    result = get_nearest_vegetation_height(lat, lon, max_distance_km=10.0)
    
    if result:
        print(f"✓ Found measurement:")
        print(f"  Vegetation height: {result['vegetation_height_m']:.2f} meters")
        print(f"  Ground elevation: {result['ground_elevation_m']:.2f} meters")
        print(f"  Canopy elevation: {result['canopy_elevation_m']:.2f} meters")
        #print(f"  Distance from query: {result['distance_km']:.2f} km")
        print(f"  Date: {result['date']}")
        print(f"  Data source: {result['data_source']}")
    else:
        print("✗ No ICESat-2 data found within 10 km")
        print("  Suggestions:")
        print("  - Try increasing max_distance_km to 20 or 50")
        print("  - Expand date range to include more data")
        print("  - Check if location is in a frequently cloudy area")
    
    print()
    
    # Method 2: Get statistics from multiple measurements
    print("Method 2: Statistics from Multiple Measurements")
    print("-" * 70)
    stats = get_vegetation_statistics(lat, lon, buffer_km=5.0)
    
    if stats:
        print(f"✓ Found {stats['num_measurements']} measurements within {stats['search_radius_km']} km:")
        print(f"  Mean height: {stats['mean_height_m']:.2f} ± {stats['std_dev_m']:.2f} meters")
        print(f"  Median height: {stats['median_height_m']:.2f} meters")
        print(f"  Range: {stats['min_height_m']:.2f} - {stats['max_height_m']:.2f} meters")
    else:
        print("✗ No ICESat-2 data found")
    
    print()
    
    # Method 3: Save to CSV
    print("Method 3: Save Results to CSV")
    print("-" * 70)
    save_results_to_csv(lat, lon, buffer_km=10.0, 
                       output_file="fort_dick_vegetation.csv")
    
    print()
    print("=" * 70)
    print("NOTES")
    print("=" * 70)
    print("""
ICESat-2 via SlideRule:
  ✓ Very current data (updated every 91 days)
  ✓ Easy to use (no NASA EarthData login required)
  ✓ Cloud processing (fast)
  ✓ Good vertical accuracy
  
Limitations:
  ✗ Sparse spatial coverage (along tracks only)
  ✗ ~3km between tracks
  ✗ May have gaps due to clouds
  
Best for: Point measurements, time series, change detection
Not ideal for: Complete area mapping
    """)


if __name__ == '__main__':
    main()