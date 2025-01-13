import math
import random
from enum import Enum
from dataclasses import dataclass
from itertools import permutations
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

    def _get_route_cost(self, start: int, end: int, criteria_weights: list[CriteriaWeight]) -> float:
        """Calculate the cost of traveling from start to end based on criteria weights."""
        weights = {
            Criteria.DISTANCE: self.matrix[start][end][0],
            Criteria.TIME: self.matrix[start][end][1],
            Criteria.COST: self.matrix[start][end][2]
        }
        return sum([weights[weight.criteria] * weight.weight for weight in criteria_weights])

    def _get_path_metrics(self, path: list[int]) -> tuple[float, float, float]:
        """Calculate total distance, duration, and cost of a path."""
        total_distance = 0
        total_duration = 0
        total_cost = 0
        
        for i in range(len(path) - 1):
            distance, duration, cost = self.matrix[self.cities.index(path[i])][self.cities.index(path[i + 1])]
            total_distance += distance
            total_duration += duration
            total_cost += cost
            
        return total_distance, total_duration, total_cost

    def optimize_route_brute_force(self, 
                                start: str, 
                                destinations: list[str] = None, 
                                end: str = None, 
                                criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS) -> tuple[list[str], float, float, float]:
        """
        Plan the optimal route using a brute-force algorithm.
        
        Args:
            start: Starting city
            destinations: List of cities to visit (if None, visit all cities)
            end: End city (if None, end at the last unvisited city)
            criteria_weights: Weights for distance, time, and cost optimization
            
        Returns:
            Tuple containing:
            - List of cities in optimal order
            - Total distance in km
            - Total duration in hours
            - Total cost in euros
        """
        if not sum([weight.weight for weight in criteria_weights]) == 1:
            raise ValueError('Criteria weights must sum up to 1')
        
        # Set up destinations
        start = [start]
        end = [end] if end else []
        destinations = set(destinations) - set(start) - set(end) if set(destinations) else set(self.cities) - set(start) - set(end)

        # Initialize starting path
        best_cost = float('inf')
        best_path = None

        # Generate all permutations of destinations
        for perm in permutations(destinations):
            path = start + list(perm) + end
            total_cost = sum([self._get_route_cost(self.cities.index(path[i]), self.cities.index(path[i + 1]), criteria_weights) for i in range(len(path) - 1)])
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path

        # Calculate metrics
        total_distance, total_duration, total_cost = self._get_path_metrics(best_path)

        return best_path, total_distance, total_duration, total_cost
        
    def optimize_route_greedy(self, 
                            start: str, 
                            destinations: list[str] = None, 
                            end: str = None, 
                            criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS) -> tuple[list[str], float, float, float]:
        """
        Plan the optimal route using a greedy algorithm.
        
        Args:
            start: Starting city
            destinations: List of cities to visit (if None, visit all cities)
            end: End city (if None, end at the last unvisited city)
            criteria_weights: Weights for distance, time, and cost optimization
            
        Returns:
            Tuple containing:
            - List of cities in optimal order
            - Total distance in km
            - Total duration in hours
            - Total cost in euros
        """
        if not sum([weight.weight for weight in criteria_weights]) == 1:
            raise ValueError('Criteria weights must sum up to 1')

        # Initialize metrics
        total_distance = 0
        total_duration = 0
        total_cost = 0
        
        # Set up start/end indices and destinations
        start_index = self.cities.index(start)
        end_index = self.cities.index(end) if end else start_index
        destinations = set(destinations) | {start, self.cities[end_index]} if set(destinations) else set(self.cities)

        # Track visited cities
        visited = [self.cities[i] not in destinations or i == start_index or i == end_index 
                  for i in range(len(self.cities))]
        
        # Build path
        path = [start]
        current_index = start_index
        
        while len(path) < len(destinations) - 1:
            # Find the unvisited city with lowest cost
            best_cost = float('inf')
            best_city_index = None
            
            # Check each possible city
            for city_index in range(len(self.cities)):
                if visited[city_index]:
                    continue
                    
                # Calculate cost to reach this city
                cost = self._get_route_cost(current_index, city_index, criteria_weights)
                
                # Update best if this is better
                if cost < best_cost:
                    best_cost = cost
                    best_city_index = city_index
                    
            next_index = best_city_index

            # Update metrics
            distance, duration, cost = self.matrix[current_index][next_index]
            total_distance += distance
            total_duration += duration
            total_cost += cost

            # Update state
            visited[next_index] = True
            path.append(self.cities[next_index])
            current_index = next_index

        # Handle final destination
        final_city = end or self.cities[visited.index(False)]
        path.append(final_city)
        
        # Add final leg metrics
        end_index = self.cities.index(final_city)
        distance, duration, cost = self.matrix[current_index][end_index]
        total_distance += distance
        total_duration += duration
        total_cost += cost

        return path, total_distance, total_duration, total_cost

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
        return optimized_route, best_metrics[0], best_metrics[1], best_metrics[2]