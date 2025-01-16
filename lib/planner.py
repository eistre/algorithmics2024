import math
import random
import heapq
from enum import Enum
from dataclasses import dataclass
from itertools import permutations
from lib.location import get_city_coordinates, get_distance_time_cost_matrix
from lib.sample_loader import load_cities_with_coordinates, load_distance_time_cost_matrix

class Criteria(Enum):
    """Criteria for route optimization."""
    DISTANCE = 'distance'
    DURATION = 'duration'
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
    CriteriaWeight(Criteria.DURATION, 0),
    CriteriaWeight(Criteria.COST, 0)
]
        
class TravelPlanner:
    """Graph of cities and routes between them."""
    def __init__(self, cities: list[str] = [], load_from_file: bool = False):
        self.cities = cities
        self.algorithm_map = {
            'brute_force': self.optimize_route_brute_force,
            'greedy': self.optimize_route_greedy,
            'simulated_annealing': self.optimize_route_simulated_annealing,
            'floyd_warshall': self.optimize_route_floyd_warshall,
            'dijkstra': self.optimize_route_dijkstra,
            'a_star': self.optimize_route_a_star
        }

        if load_from_file:
            self.coordinates = load_cities_with_coordinates()
            self.cities = [city for city, _, _ in self.coordinates]
            self.matrix = load_distance_time_cost_matrix()
        else:
            self.coordinates: list[tuple[str, float, float]] = [get_city_coordinates(city) for city in cities]
            self.matrix: list[list[tuple[float, float, float]]] = get_distance_time_cost_matrix(self.coordinates)

        self.city_indices = {city: i for i, city in enumerate(self.cities)}

    def _get_route_cost(self, start: int, end: int, criteria_weights: list[CriteriaWeight]) -> float:
        """Calculate the cost of traveling from start to end based on criteria weights."""
        weights = {
            Criteria.DISTANCE: self.matrix[start][end][0],
            Criteria.DURATION: self.matrix[start][end][1],
            Criteria.COST: self.matrix[start][end][2]
        }
        return sum([weights[weight.criteria] * weight.weight for weight in criteria_weights])

    def _get_path_metrics(self, path: list[int]) -> tuple[float, float, float]:
        """Calculate total distance, duration, and cost of a path."""
        total_distance = 0
        total_duration = 0
        total_cost = 0
        
        for i in range(len(path) - 1):
            distance, duration, cost = self.matrix[self.city_indices[path[i]]][self.city_indices[path[i + 1]]]
            total_distance += distance
            total_duration += duration
            total_cost += cost
            
        return total_distance, total_duration, total_cost

    def optimize_route(self, start: str, destinations: list[str] = [],
                    end: str = None, criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS,
                    algorithm: str = 'greedy') -> tuple[list[str], float, float, float]:
        """Plan the optimal route using the specified algorithm."""
        

        if algorithm not in self.algorithm_map:
            raise ValueError('Invalid algorithm')
        
        return self.algorithm_map[algorithm](start, destinations, end, criteria_weights)

    def optimize_route_brute_force(self, 
                                start: str, 
                                destinations: list[str] = [], 
                                end: str = None, 
                                criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS) -> tuple[list[str], float, float, float]:
        """
        Plan the optimal route using a brute-force algorithm.
        
        Args:
            start: Starting city
            destinations: List of cities to visit (if None, visit all cities)
            end: Ending city (optional, defaults to start if None)
            criteria_weights: Weights for distance, time, and cost optimization
            
        Returns:
            Tuple containing:
            - List of cities in optimal order
            - Total distance in km
            - Total duration in hours
            - Total cost in euros
        """
        if not math.isclose(sum(weight.weight for weight in criteria_weights), 1.0, rel_tol=1e-9):
            raise ValueError('Criteria weights must sum up to 1')
        
        # Set up destinations
        destinations = set(destinations) - {start, end} or set(self.cities) - {start, end}

        # Convert start and end to lists
        start = [start]
        end = [end] if end else start  # Return to start if no end specified

        # Initialize starting path
        best_cost = float('inf')
        best_path = None

        # Generate all permutations of destinations
        for perm in permutations(destinations):
            path = start + list(perm) + end
            total_cost = sum([self._get_route_cost(self.city_indices[path[i]], self.city_indices[path[i + 1]], criteria_weights) for i in range(len(path) - 1)])
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path

        # Calculate metrics
        total_distance, total_duration, total_cost = self._get_path_metrics(best_path)

        return best_path, total_distance, total_duration, total_cost
        
    def optimize_route_greedy(self, 
                            start: str, 
                            destinations: list[str] = [], 
                            end: str = None, 
                            criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS) -> tuple[list[str], float, float, float]:
        """
        Plan the optimal route using a greedy algorithm.

        Args:
            start: Starting city
            destinations: List of cities to visit (if None, visit all cities)
            end: Ending city (optional, defaults to start if None)
            criteria_weights: Weights for distance, time, and cost optimization

        Returns:
            Tuple containing:
            - List of cities in optimal order
            - Total distance in km
            - Total duration in hours
            - Total cost in euros
        """
        # Validate criteria weights
        if not math.isclose(sum(weight.weight for weight in criteria_weights), 1.0, rel_tol=1e-9):
            raise ValueError('Criteria weights must sum up to 1')
        
        # Initialize destinations
        destinations = set(destinations) - {start, end} or set(self.cities) - {start, end}

        # Initialize current city
        current_city = start
        path = [current_city]

        # Greedy algorithm
        while destinations:
            # Find the closest next city
            current_city_index = self.city_indices[current_city]
            next_destination = min(
                destinations,
                key=lambda city: self._get_route_cost(current_city_index, self.city_indices[city], criteria_weights)
            )

            # Update state
            path.append(next_destination)
            current_city = next_destination
            destinations.remove(next_destination)

        # Add final city (either end or return to start)
        final_city = end if end else start
        path.append(final_city)

        # Calculate metrics
        total_distance, total_duration, total_cost = self._get_path_metrics(path)

        return path, total_distance, total_duration, total_cost

    def optimize_route_simulated_annealing(self, start: str, destinations: list[str] = None, 
                                    end: str = None, 
                                    criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS,
                                    initial_temperature: float = 1000,
                                    cooling_rate: float = 0.99,
                                    iterations_per_temp: int = 100) -> tuple[list[str], float, float, float]:
        """
        Plan the optimal route using simulated annealing algorithm.
        Always starts from the given start city.
        """
        # Validate criteria weights
        if not math.isclose(sum(weight.weight for weight in criteria_weights), 1.0, rel_tol=1e-9):
            raise ValueError('Criteria weights must sum up to 1')

        # Initialize destinations
        if destinations is None:
            destinations = list(set(self.cities) - {start})
        else:
            destinations = list(set(destinations) - {start})

        end = end if end else start
        start_idx = self.city_indices[start]
        end_idx = self.city_indices[end]

        def get_neighbor(current_route: list[int]) -> list[int]:
            """Generate a neighbor solution by swapping two random intermediate cities."""
            if len(current_route) <= 3:
                return current_route.copy()
                
            neighbor = current_route.copy()
            # Only swap intermediate cities - NEVER touch index 0 (start city)
            pos1, pos2 = random.sample(range(1, len(neighbor) - 1), 2)
            neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
            return neighbor

        def calculate_route_score(route: list[int]) -> tuple[float, float, float, float]:
            """Calculate total score, distance, duration, and cost for a route."""
            total_distance = total_duration = total_cost = 0
            
            for i in range(len(route) - 1):
                total_distance += self.matrix[route[i]][route[i + 1]][0]
                total_duration += self.matrix[route[i]][route[i + 1]][1]
                total_cost += self.matrix[route[i]][route[i + 1]][2]
                
            total_score = (
                total_distance * criteria_weights[0].weight +
                total_duration * criteria_weights[1].weight +
                total_cost * criteria_weights[2].weight
            )
            
            return total_score, total_distance, total_duration, total_cost

        # Create initial solution - ALWAYS start with start_idx
        dest_indices = [self.city_indices[city] for city in destinations]
        random.shuffle(dest_indices)
        current_solution = [start_idx] + dest_indices + [end_idx]
        
        current_score, current_distance, current_duration, current_cost = calculate_route_score(current_solution)
        best_solution = current_solution.copy()
        best_score = current_score
        best_metrics = (current_distance, current_duration, current_cost)

        # Simulated annealing process
        temperature = initial_temperature
        while temperature > 1:
            for _ in range(iterations_per_temp):
                # get_neighbor NEVER modifies the start city
                neighbor = get_neighbor(current_solution)
                new_score, new_distance, new_duration, new_cost = calculate_route_score(neighbor)
                
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
    
    def optimize_route_a_star(self, start: str, destinations: list[str] = None, 
                         end: str = None,
                         criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS) -> tuple[list[str], float, float, float]:
        """
        Plan the optimal route using A* algorithm.
        
        Args:
            start: Starting city
            destinations: List of cities to visit (optional)
            end: Ending city (optional, defaults to start if None)
            criteria_weights: List of weights for distance, duration, and cost
        
        Returns:
            Tuple containing:
            - List of cities in optimized order
            - Total distance
            - Total duration
            - Total cost
        """
        # Validate criteria weights
        if not math.isclose(sum(weight.weight for weight in criteria_weights), 1.0, rel_tol=1e-9):
            raise ValueError('Criteria weights must sum up to 1')

        # Initialize cities to visit
        if not destinations:
            destinations = set(self.cities)
        else:
            destinations = set(destinations) | {start}
            if end:
                destinations.add(end)
        
        end_index = self.city_indices[end] if end else self.city_indices[start]
        unvisited = set(i for i, city in enumerate(self.cities) if city in destinations)
        start_index = self.city_indices[start]
        
        def calculate_edge_cost(from_idx: int, to_idx: int) -> float:
            """Calculate weighted cost between two cities."""
            distance = self.matrix[from_idx][to_idx][0]
            duration = self.matrix[from_idx][to_idx][1]
            cost = self.matrix[from_idx][to_idx][2]
            
            return (distance * criteria_weights[0].weight +
                    duration * criteria_weights[1].weight +
                    cost * criteria_weights[2].weight)
        
        def heuristic(current_idx: int, remaining: set[int]) -> float:
            if not remaining:
                return calculate_edge_cost(current_idx, end_index)
            return min(calculate_edge_cost(current_idx, city) for city in remaining) + \
                calculate_edge_cost(min(remaining, key=lambda x: calculate_edge_cost(x, end_index)), end_index)

        class Node:
            def __init__(self, city_idx: int, path: list[int], 
                        unvisited: set[int], g_score: float):
                self.city_idx = city_idx
                self.path = path
                self.unvisited = unvisited
                self.g_score = g_score
                self.h_score = heuristic(city_idx, unvisited)
                self.f_score = g_score + self.h_score
            
            def __lt__(self, other):
                return self.f_score < other.f_score

        # Initialize priority queue with start node
        start_node = Node(start_index, [start_index], 
                        unvisited - {start_index}, 0)
        queue = [start_node]
        heapq.heapify(queue)
        
        while queue:
            current = heapq.heappop(queue)
            
            # If all cities visited, find path to end
            if not current.unvisited and current.city_idx == end_index:
                # Calculate final metrics
                total_distance = total_duration = total_cost = 0
                path = current.path
                
                for i in range(len(path) - 1):
                    from_idx = path[i]
                    to_idx = path[i + 1]
                    total_distance += self.matrix[from_idx][to_idx][0]
                    total_duration += self.matrix[from_idx][to_idx][1]
                    total_cost += self.matrix[from_idx][to_idx][2]
                
                return ([self.cities[i] for i in path],
                        total_distance, total_duration, total_cost)
            
            # Generate successor nodes
            for next_city in current.unvisited | {end_index}:
                # Calculate actual cost to reach next_city
                new_g_score = (current.g_score + 
                            calculate_edge_cost(current.city_idx, next_city))
                
                new_path = current.path + [next_city]
                new_unvisited = current.unvisited - {next_city}
                
                # Only add end_city if all others are visited
                if next_city == end_index and new_unvisited:
                    continue
                    
                new_node = Node(next_city, new_path, new_unvisited, new_g_score)
                heapq.heappush(queue, new_node)
        
        raise ValueError("No valid route found")
    
    def optimize_route_floyd_warshall(self,
                                    start: str,
                                    destinations: list[str] = [],
                                    end: str = None,
                                    criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS) -> tuple[list[str], float, float, float]:
        """
        Plan the optimal route using the Floyd-Warshall algorithm.

        Args:
            start: Starting city
            destinations: List of cities to visit (if None, visit all cities)
            end: Ending city (optional, defaults to start if None)
            criteria_weights: Weights for distance, time, and cost optimization

        Returns:
            Tuple containing:
            - List of cities in optimal order
            - Total distance in km
            - Total duration in hours
            - Total cost in euros
        """
        # Validate criteria weights
        if not math.isclose(sum(weight.weight for weight in criteria_weights), 1.0, rel_tol=1e-9):
            raise ValueError('Criteria weights must sum up to 1')
        
        # Initialize destinations
        destinations = set(destinations) - {start, end} or set(self.cities) - {start, end}

        # Initialize distance and next city matrices
        n = len(self.cities)
        distance_matrix = [[float('inf')] * n for _ in range(n)]
        next_city_matrix = [[None] * n for _ in range(n)]

        # Initialize matrix with direct distances
        for i in range(n):
            distance_matrix[i][i] = 0
            for j in range(n):
                distance_matrix[i][j] = self._get_route_cost(i, j, criteria_weights)
                next_city_matrix[i][j] = j

        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                        distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                        next_city_matrix[i][j] = next_city_matrix[i][k]

        # Optimize route according to new matrix with greedy algorithm
        current_city = start
        path = [current_city]

        while destinations:
            # Find the closest next city
            current_city_index = self.city_indices[current_city]
            next_destination = min(
                destinations,
                key=lambda city: distance_matrix[current_city_index][self.city_indices[city]]
            )
            next_destination_index = self.city_indices[next_destination]

            # Reconstruct path
            while current_city_index != next_destination_index:
                current_city_index = next_city_matrix[current_city_index][next_destination_index]

                # If a passed city is in the destinations, remove it
                if self.cities[current_city_index] in destinations:
                    destinations.remove(self.cities[current_city_index])
                
                path.append(self.cities[current_city_index])

            # Update state
            current_city = next_destination

        # Add final city (either end or return to start)
        final_city = end if end else start
        path.append(final_city)

        # Calculate metrics
        total_distance, total_duration, total_cost = self._get_path_metrics(path)

        return path, total_distance, total_duration, total_cost

    def optimize_route_dijkstra(self,
                                start: str,
                                destinations: list[str] = [],
                                end: str = None,
                                criteria_weights: list[CriteriaWeight] = DEFAULT_CRITERIA_WEIGHTS) -> tuple[list[str], float, float, float]:
        """
        Plan the optimal route using Dijkstra's algorithm.

        Args:
            start: Starting city
            destinations: List of cities to visit (if None, visit all cities)
            end: Ending city (optional, defaults to start if None)
            criteria_weights: Weights for distance, time, and cost optimization

        Returns:
            Tuple containing:
            - List of cities in optimal order
            - Total distance in km
            - Total duration in hours
            - Total cost in euros
        """
        # Validate criteria weights
        if not math.isclose(sum(weight.weight for weight in criteria_weights), 1.0, rel_tol=1e-9):
            raise ValueError('Criteria weights must sum up to 1')
        
        # Initialize destinations
        destinations = set(destinations) - {start, end} or set(self.cities) - {start, end}

        # Dijkstra's algorithm
        def dijkstra(source_idx: int) -> tuple[list[float], list[int]]:
            """Find the shortest path from the source to all other nodes."""
            n = len(self.cities)
            distance = [float('inf')] * n
            prev = [None] * n
            visited = [False] * n

            distance[source_idx] = 0

            for _ in range(n):
                # Find the vertex with the minimum distance
                min_distance = float('inf')
                min_idx = -1

                for i in range(n):
                    if not visited[i] and distance[i] < min_distance:
                        min_distance = distance[i]
                        min_idx = i

                visited[min_idx] = True

                # Update distances
                for i in range(n):
                    if not visited[i]:
                        new_distance = distance[min_idx] + self._get_route_cost(min_idx, i, criteria_weights)
                        if new_distance < distance[i]:
                            distance[i] = new_distance
                            prev[i] = min_idx

            return distance, prev
        
        # Build the path using dijkstra's algorithm
        current_city = start
        path = [current_city]

        while destinations:
            # Get shortest path from current city
            current_city_index = self.city_indices[current_city]
            distances, previous = dijkstra(current_city_index)

            # Find the closest next city
            next_destination = min(
                destinations,
                key=lambda city: distances[self.city_indices[city]]
            )
            next_destination_index = self.city_indices[next_destination]

            # Reconstruct path
            temp_path = []
            while next_destination_index is not None:
                city = self.cities[next_destination_index]

                # If a passed city is in the destinations, remove it
                if city in destinations:
                    destinations.remove(city)

                temp_path.append(city)
                next_destination_index = previous[next_destination_index]

            # Update state
            path += temp_path[::-1][1:]
            current_city = next_destination
        
        # Add final city (either end or return to start)
        final_city = end if end else start
        path.append(final_city)

        # Calculate metrics
        total_distance, total_duration, total_cost = self._get_path_metrics(path)

        return path, total_distance, total_duration, total_cost
    