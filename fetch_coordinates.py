import time
from geopy.geocoders import Nominatim
import json

# List of locations in Mumbai to find coordinates for
locations = [
    'Worli', 'Colaba', 'Andheri', 'Bandra', 'Borivali', 
    'Chembur', 'Kurla', 'Powai', 'Sion', 'Mulund', 
    'Malad', 'Dadar', 'Dharavi', 'Juhu', 'Malabar Hill', 
    'Navi Mumbai', 'Thane', 'Kalyan', 'Vasai', 'Virar', 'Vile Parle'
]

current_coordinates = {
    'Worli': (19.018227, 72.816833),
    'Colaba': (18.910000, 72.809998),
    'Andheri': (19.116625, 72.862358),
    'Bandra': (19.05444444, 72.84055556),
    'Borivali': (19.228825, 72.854118),
    'Chembur': (19.032801, 72.896355),
    'Kurla': (19.059984, 72.889999),
    'Powai': (19.11640000, 72.90471000),
    'Sion': (19.0390, 72.8619),
    'Mulund': (19.175547, 72.972099),
    'Malad': (19.18611111, 72.84861111),
    'Dadar': (19.019908, 72.841330),
    'Dharavi': (19.05000000, 72.86667000),
    'Juhu': (19.10000000, 72.83000000),
    'Malabar Hill': (18.95000000, 72.79500000),
    'Navi Mumbai': (19.03681000, 73.01582000),
    'Thane': (19.2183, 72.9780),
    'Kalyan': (19.2403, 73.1305),
    'Vasai': (19.3919, 72.8397),
    'Virar': (19.4564, 72.8110),
    'Vile Parle': (19.0969, 72.8497)
}

def fetch_coordinates():
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="mumbai_location_finder")
    
    new_coordinates = {}
    
    for location in locations:
        try:
            # Add Mumbai to the query for better results
            query = f"{location}, Mumbai, India"
            
            # For locations that are already cities/regions
            if location in ["Navi Mumbai", "Thane", "Kalyan", "Vasai", "Virar"]:
                query = f"{location}, Maharashtra, India"
                
            print(f"Fetching coordinates for: {query}")
            
            # Get location data
            location_data = geolocator.geocode(query)
            
            if location_data:
                # Store coordinates
                new_coordinates[location] = (location_data.latitude, location_data.longitude)
                print(f"Found: {location} - {new_coordinates[location]}")
            else:
                print(f"Could not find coordinates for {location}, using existing data if available")
                if location in current_coordinates:
                    new_coordinates[location] = current_coordinates[location]
                    print(f"Using existing data: {location} - {new_coordinates[location]}")
            
            # Sleep to respect API rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching coordinates for {location}: {e}")
            if location in current_coordinates:
                new_coordinates[location] = current_coordinates[location]
                print(f"Using existing data: {location} - {new_coordinates[location]}")
            time.sleep(1)
    
    return new_coordinates

def generate_json_snippet(coordinates):
    """Generate JSON for location coordinates"""
    import json
    
    # Convert tuple coordinates to lists for JSON serialization
    json_data = {location: [lat, lng] for location, (lat, lng) in coordinates.items()}
    
    # Return formatted JSON string with indentation
    return json.dumps(json_data, indent=2)

if __name__ == "__main__":
    print("Fetching coordinates from OpenStreetMap (Nominatim API)...")
    new_coordinates = fetch_coordinates()
    
    # Generate and print the updated code
    print("\nUpdated location coordinates in JSON format:")
    updated_json = generate_json_snippet(new_coordinates)
    print(updated_json)
    
    # Save as JSON for reference
    with open("location_coordinates.json", "w") as f:
        json.dump({k: list(v) for k, v in new_coordinates.items()}, f, indent=2)
    
    print("Coordinates saved as JSON in 'location_coordinates.json'")
