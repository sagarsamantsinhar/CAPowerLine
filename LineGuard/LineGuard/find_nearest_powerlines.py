"""
Enhanced debug to find nearest power lines and suggest solutions.
"""

import json
from math import radians, cos, sin, asin, sqrt


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in miles."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 3959 * c


def find_nearest_powerlines(geojson_file, center_lat, center_lon, max_search_radius=200):
    """Find the nearest power lines and suggest solutions."""
    
    print(f"╔═══════════════════════════════════════════════════════════╗")
    print(f"║     HIFLD Power Line Analysis for Fort Dick, CA          ║")
    print(f"╚═══════════════════════════════════════════════════════════╝")
    print()
    print(f"Search Center: {center_lat}°N, {abs(center_lon)}°W")
    print(f"Max Search Radius: {max_search_radius} miles")
    print()
    
    # Load GeoJSON
    try:
        with open(geojson_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded GeoJSON with {len(data.get('features', []))} features")
    except FileNotFoundError:
        print(f"✗ ERROR: Cannot find '{geojson_file}'")
        print(f"\nPlease download HIFLD Electric Power Transmission Lines from:")
        print(f"https://hifld-geoplatform.opendata.arcgis.com/datasets/electric-power-transmission-lines")
        return
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return
    
    print()
    print("Searching for nearest power lines...")
    print()
    
    # Find nearest points
    nearest_lines = []
    
    for idx, feature in enumerate(data.get('features', [])):
        if 'geometry' not in feature or not feature['geometry']:
            continue
        
        geom = feature['geometry']
        if geom['type'] not in ['LineString', 'MultiLineString']:
            continue
        
        # Get coordinates
        if geom['type'] == 'LineString':
            coord_lists = [geom['coordinates']]
        else:  # MultiLineString
            coord_lists = geom['coordinates']
        
        # Find minimum distance from this line to center
        min_dist = float('inf')
        closest_point = None
        
        for coords in coord_lists:
            for lon, lat in coords:
                dist = haversine_distance(center_lat, center_lon, lat, lon)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (lat, lon)
        
        if min_dist <= max_search_radius:
            # Get line properties
            props = feature.get('properties', {})
            voltage = props.get('VOLTAGE', props.get('voltage', 'Unknown'))
            owner = props.get('OWNER', props.get('owner', 'Unknown'))
            line_type = props.get('TYPE', props.get('type', 'Unknown'))
            
            nearest_lines.append({
                'distance': min_dist,
                'point': closest_point,
                'voltage': voltage,
                'owner': owner,
                'type': line_type,
                'feature_idx': idx
            })
    
    # Sort by distance
    nearest_lines.sort(key=lambda x: x['distance'])
    
    if not nearest_lines:
        print(f"✗ NO POWER LINES found within {max_search_radius} miles!")
        print()
        print("This could mean:")
        print("  1. The GeoJSON file doesn't cover this region")
        print("  2. There are genuinely no transmission lines in this remote coastal area")
        print("  3. The file only contains high-voltage lines (smaller lines excluded)")
        return
    
    # Display results
    print(f"✓ Found {len(nearest_lines)} power line(s) within {max_search_radius} miles")
    print()
    print("═" * 70)
    print("NEAREST POWER LINES:")
    print("═" * 70)
    
    for i, line in enumerate(nearest_lines[:10]):  # Show top 10
        print(f"\n#{i+1} - Distance: {line['distance']:.2f} miles")
        print(f"    Location: {line['point'][0]:.6f}°N, {abs(line['point'][1]):.6f}°W")
        print(f"    Voltage: {line['voltage']}")
        print(f"    Owner: {line['owner']}")
        print(f"    Type: {line['type']}")
    
    print()
    print("═" * 70)
    print("RECOMMENDATIONS:")
    print("═" * 70)
    
    closest = nearest_lines[0]
    
    if closest['distance'] > 50:
        print(f"\n⚠ The nearest power line is {closest['distance']:.1f} miles away!")
        print(f"\n  OPTION 1: Increase your search radius")
        print(f"    Change radius_miles to at least: {int(closest['distance']) + 10}")
        print(f"\n  OPTION 2: Move search center inland")
        print(f"    Try center point: ({closest['point'][0]:.6f}, {closest['point'][1]:.6f})")
        print(f"    This is near the closest power line.")
    else:
        print(f"\n✓ Nearest line is {closest['distance']:.1f} miles away (within 50 mile radius)")
        print(f"\n  Your original parameters should work!")
        print(f"  There may be an issue with the coordinate projection or sampling.")
    
    print()
    
    # Generate sample code
    print("═" * 70)
    print("SUGGESTED CODE:")
    print("═" * 70)
    print(f"""
# Use increased radius
points = extract_powerline_points(
    geojson_file='{geojson_file}',
    center_lat={center_lat},
    center_lon={center_lon},
    radius_miles={max(50, int(closest['distance']) + 10)},
    interval_feet=3
)

# OR use the nearest power line location as center
points = extract_powerline_points(
    geojson_file='{geojson_file}',
    center_lat={closest['point'][0]:.6f},
    center_lon={closest['point'][1]:.6f},
    radius_miles=50,
    interval_feet=3
)
""")


if __name__ == '__main__':
    # Your parameters
    geojson_file = "resources/ustrlines.geojson"
    center_lat = 41.868074
    center_lon = -124.152736
    print("in main")
    find_nearest_powerlines(geojson_file, center_lat, center_lon, max_search_radius=200)