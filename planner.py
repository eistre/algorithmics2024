from enum import Enum
from dataclasses import dataclass
from location import get_city_coordinates, get_distance_time_cost_matrix
from sample_loader import load_cities_with_coordinates, load_distance_time_cost_matrix
import math
import random

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

        # Update the total distance, duration, and cost
        total_distance += self.matrix[current_index][end_index][0]
        total_duration += self.matrix[current_index][end_index][1]
        total_cost += self.matrix[current_index][end_index][2]
        
        return path, total_distance, total_duration/60, total_cost

    def optimize_route_simulated_annealing(self, start: str, destinations: list[str] = None, 
                                        end: str = None, 
                                        criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS,
                                        initial_temperature: float = 1000,
                                        cooling_rate: float = 0.99,
                                        iterations_per_temp: int = 100) -> tuple[list[str], float, float, float]:
        """
        Plan the optimal route using simulated annealing algorithm.
        """
        # Validate criteria weights
        if not sum([weight.weight for weight in criteria_weights]) == 1:
            raise ValueError('Criteria weights must sum up to 1')

        # Initialize cities to visit
        if not destinations:
            destinations = set(self.cities)
        else:
            destinations = set(destinations) | {start}
            if end:
                destinations.add(end)

        # Set up start and end indices
        start_index = self.cities.index(start)
        end_index = self.cities.index(end) if end else start_index

        def calculate_route_score(route: list[int]) -> tuple[float, float, float, float]:
            """Calculate total score, distance, duration, and cost for a route."""
            total_distance = total_duration = total_cost = 0
            
            for i in range(len(route) - 1):
                current_city = route[i]
                next_city = route[i + 1]
                
                # Add up individual metrics
                total_distance += self.matrix[current_city][next_city][0]
                total_duration += self.matrix[current_city][next_city][1]
                total_cost += self.matrix[current_city][next_city][2]
                
            # Calculate weighted score
            total_score = (
                total_distance * criteria_weights[0].weight +
                total_duration * criteria_weights[1].weight +
                total_cost * criteria_weights[2].weight
            )
            
            return total_score, total_distance, total_duration, total_cost

        def get_neighbor(current_route: list[int]) -> list[int]:
            """Generate a neighbor solution by swapping two random cities."""
            # Don't modify start and end positions
            if len(current_route) <= 3:  # Not enough cities to swap
                return current_route.copy()
                
            neighbor = current_route.copy()
            # Select two random positions (excluding start and end cities)
            pos1, pos2 = random.sample(range(1, len(neighbor) - 1), 2)
            neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
            return neighbor

        # Create initial solution
        city_indices = [i for i in range(len(self.cities)) if self.cities[i] in destinations]
        if start_index in city_indices and start_index != city_indices[0]:
            city_indices.remove(start_index)
            city_indices.insert(0, start_index)
        if end_index in city_indices and end_index != city_indices[-1]:
            city_indices.remove(end_index)
            city_indices.append(end_index)

        current_solution = city_indices
        current_score, current_distance, current_duration, current_cost = calculate_route_score(current_solution)
        best_solution = current_solution.copy()
        best_score = current_score
        best_metrics = (current_distance, current_duration, current_cost)

        # Simulated annealing process
        temperature = initial_temperature
        
        while temperature > 1:
            for _ in range(iterations_per_temp):
                neighbor = get_neighbor(current_solution)
                new_score, new_distance, new_duration, new_cost = calculate_route_score(neighbor)
                
                # Calculate acceptance probability
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_solution = neighbor
                    current_score = new_score
                    
                    if new_score < best_score:
                        best_solution = neighbor.copy()
                        best_score = new_score
                        best_metrics = (new_distance, new_duration, new_cost)
            
            temperature *= cooling_rate

        # Convert indices back to city names
        optimized_route = [self.cities[i] for i in best_solution]
        return optimized_route, best_metrics[0], best_metrics[1]/60, best_metrics[2]