import asyncio
import aiohttp
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

async def get_route_details(source: Tuple[float, float], destination: Tuple[float, float]) -> Tuple[float, float, float]:
    url = f'http://router.project-osrm.org/route/v1/driving/{source[1]},{source[0]};{destination[1]},{destination[0]}?overview=false'

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f'Error getting route details: {response.status}')
                
                data = await response.json()
                if data['code'] != 'Ok' or not data['routes']:
                    raise Exception(f'Error getting route details')
                
                route = data['routes'][0]

                # Convert the distance from meters to kilometers
                distance = route['distance'] / 1000

                # Convert the duration from seconds to hours
                time = route['duration'] / 3600

                # Assume an average cost of 1.5 euros per litre of fuel
                # Assume an average fuel consumption of 8 litres per 100 km
                # Calculate the cost based on the distance and fuel consumption
                cost = distance * 1.5 * 8 / 100

                return distance, time, cost
    except Exception as e:
        print(f"Error getting route details: {e}")
        return None
