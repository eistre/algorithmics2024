from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Dict
from location import get_locations, get_route_details

class Criteria(Enum):
    """Criteria for route optimization."""
    DISTANCE = 'distance'
    TIME = 'time'
    COST = 'cost'

@dataclass
class CriteriaWeight:
    """Weight for a specific criteria."""
    criteria: Criteria
    weight: float       # In [0, 1]

    # Check if the weight is between 0 and 1
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError('Weight must be in [0, 1]')

@dataclass
class City:
    """City with its name and coordinates."""
    name: str
    latitude: float
    longitude: float

    # Check if the latitude and longitude are within the valid range
    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError('Latitude must be in [-90, 90]')
        if not -180 <= self.longitude <= 180:
            raise ValueError('Longitude must be in [-180, 180]')

    @property
    def coordinates(self) -> Tuple[float, float]:
        return self.latitude, self.longitude
    
@dataclass
class Route:
    """Route between two cities."""
    distance: float     # In kilometers
    time: float         # In hours
    cost: float         # In euros

    # Check if the distance, time, and cost are non-negative.
    def __post_init__(self):
        if self.distance < 0:
            raise ValueError('Distance must be non-negative')
        if self.time < 0:
            raise ValueError('Time must be non-negative')
        if self.cost < 0:
            raise ValueError('Cost must be non-negative')

    def get_weight(self, criteria_weights: List[CriteriaWeight]) -> float:
        weights = {
            Criteria.DISTANCE: self.distance,
            Criteria.TIME: self.time,
            Criteria.COST: self.cost
        }

        return sum(weight.weight * weights[weight.criteria] for weight in criteria_weights)
    
class TravelGraph:
    """Graph of cities and routes between them."""
    def __init__(self):
        self.cities: Dict[str, City] = {}
        self.routes: Dict[str, Dict[str, Route]] = {}

    async def create(self, cities: List[str]) -> "TravelGraph":
        locations = await get_locations(cities)

        for city, lat, lon in locations:
            self.cities[city] = City(city, lat, lon)
            self.routes[city] = {}

        for source in self.cities.values():
            for destination in self.cities.values():
                if source == destination:
                    continue

                # Check if an inverted route already exists
                if destination.name in self.routes and source.name in self.routes[destination.name]:
                    self.routes[source.name][destination.name] = self.routes[destination.name][source.name]
                    continue

                distance, time, cost = await get_route_details(source.coordinates, destination.coordinates)
                self.routes[source.name][destination.name] = Route(distance, time, cost)

        return self
