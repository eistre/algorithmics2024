import os
import openrouteservice
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_client = None

def _get_client():
    """Lazy load the OpenRouteService client whenever it's actually needed."""
    global _client
    if _client is None:
        _client = openrouteservice.Client(key=os.getenv('ORS_API_KEY'))
    return _client

def get_city_coordinates(city: str) -> Optional[tuple[str, float, float]]:
    """Fetches the coordinates for a city using the OpenRouteService API."""
    result = _get_client().pelias_search(text=city, size=1)

    if not result['features']:
        return None
    
    # ORS returns coordinates as [longitude, latitude]
    coordinates = result['features'][0]['geometry']['coordinates']
    return city, float(coordinates[1]), float(coordinates[0])

def get_distance_time_cost_matrix(cities: list[tuple[str, float, float]]) -> list[list[tuple[float, float, float]]]:
    """Fetches the distance and time matrix for a list of cities."""
    coordinates = [[lon, lat] for _, lat, lon in cities]

    matrices = _get_client.distance_matrix(
        locations=coordinates,
        profile='driving-car',
        metrics=['distance', 'duration'],
        units='km'
    )

    # Combine distances, durations and cost to single matrix of tuples
    matrix = []
    for i in range(len(cities)):
        row = []
        for j in range(len(cities)):
            # Get the distance
            distance = float(matrices['distances'][i][j])

            # Get the duration in minutes
            duration = float(matrices['durations'][i][j]) / 60

            # Assume the cost of fuel is 1.5€/L and the car consumes 7L/100km
            # Assume the cost of travel is 1€/hour
            cost = distance * 1.5 * 7 / 100 + duration / 60

            row.append((distance, duration, cost))

        matrix.append(row)

    return matrix
