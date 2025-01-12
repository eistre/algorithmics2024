from enum import Enum
from dataclasses import dataclass
from location import get_city_coordinates, get_distance_time_cost_matrix
from sample_loader import load_cities_with_coordinates, load_distance_time_cost_matrix

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
        
DEFAULT_CRITERIA_WEIGHTS = [
    CriteriaWeight(Criteria.DISTANCE, 1),
    CriteriaWeight(Criteria.TIME, 0),
    CriteriaWeight(Criteria.COST, 0)
]
        
class TravelPlanner:
    """Graph of cities and routes between them."""
    def __init__(self, cities: list[str] = [], load_from_file: bool = False):
        self.cities = cities

        if load_from_file:
            self.coordinates = load_cities_with_coordinates()
            self.cities = [city for city, _, _ in self.coordinates]
            self.matrix = load_distance_time_cost_matrix()
        else:
            self.coordinates: list[tuple[str, float, float]] = [get_city_coordinates(city) for city in cities]
            self.matrix: list[list[tuple[float, float, float]]] = get_distance_time_cost_matrix(self.coordinates)

    def optimize_route_greedy(self, start: str, destinations: list[str] = None, end: str = None, criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS) -> list[str]:
        """Plan the optimal route from start to end by only selecting the next city with the lowest score."""
        # Make sure criteria weights sum up to 1
        if not sum([weight.weight for weight in criteria_weights]) == 1:
            raise ValueError('Criteria weights must sum up to 1')
        
        total_distance = 0
        total_duration = 0
        total_cost = 0

        # Get the start and end city indices
        start_index = self.cities.index(start)
        end_index = self.cities.index(end) if end else start_index

        # Keep track of visited cities
        visited = [False] * len(self.cities)
        visited[start_index] = True
        visited[end_index] = True

        # If destinations are provided, set others as visited
        # TODO: This can probably be optimized
        if not destinations:
            destinations = set(self.cities)
        else:
            destinations = set(destinations) | {start, self.cities[end_index]}

        for i in range(len(self.cities)):
            if self.cities[i] not in destinations:
                visited[i] = True

        path = [start]
        current_index = start_index
        while len(path) < len(destinations) - 1:
            # Get the next best city to visit
            next_index = None
            next_score = float('inf')
            for i in range(len(self.cities)):
                if visited[i]:
                    continue

                # Calculate the score for the next city
                score = sum([self.matrix[current_index][i][j] * weight.weight for j, weight in enumerate(criteria_weights)])
                if score < next_score:
                    next_index = i
                    next_score = score

            # Update the total distance, duration, and cost
            total_distance += self.matrix[current_index][next_index][0]
            total_duration += self.matrix[current_index][next_index][1]
            total_cost += self.matrix[current_index][next_index][2]

            # Set the next city as visited and add it to the path
            visited[next_index] = True
            path.append(self.cities[next_index])
            current_index = next_index

        # Add the end city to the path
        if end:
            path.append(end)
        else:
            end_index = visited.index(False)
            path.append(self.cities[end_index])
        
        return path, total_distance, total_duration, total_cost
