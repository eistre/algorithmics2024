import asyncio
from typing import List, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.adapters import AioHTTPAdapter

async def get_location_for_city(city: str, locator: Nominatim) -> Optional[Tuple[str, float, float]]:
    try:
        # Get the location of the city
        location = await locator.geocode(city, timeout=10)

        if not location:
            return None
        
        # Return the city name and the latitude and longitude
        return city, location.latitude, location.longitude
    except Exception as e:
        print(f"Error getting location for {city}: {e}")
        return None

async def get_locations(cities: List[str]) -> List[Tuple[str, float, float]]:
    locations = []
    async with Nominatim(user_agent='algorithmics2024', adapter_factory=AioHTTPAdapter) as geolocator:
        tasks = [get_location_for_city(city, geolocator) for city in cities]
        results = await asyncio.gather(*tasks)

        # Filter out the None values
        locations = [loc for loc in results if loc is not None]

    return locations
