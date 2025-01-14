import time
from planner import TravelPlanner, CriteriaWeight, Criteria

def main():
    planner = TravelPlanner(load_from_file=True)
    criteria_weights = [CriteriaWeight(Criteria.DISTANCE, 0), CriteriaWeight(Criteria.DURATION, 1), CriteriaWeight(Criteria.COST, 0)]

    print('\nOptimizing route from Oslo to Barcelona: brute force')
    start = time.time()
    route, distance, duration, cost = planner.optimize_route_brute_force(start='Oslo', end=None, destinations=['Stockholm', 'Helsinki', 'Barcelona', 'Berlin', 'Vilnius', 'Tallinn', 'Copenhagen', 'Frankfurt', 'Warsaw', 'Rome'], criteria_weights=criteria_weights)
    end = time.time()

    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')
    print(f'Execution time: {end - start:.5f} seconds')

    print('\nOptimizing route from Oslo to Barcelona: greedy')
    start = time.time()
    route, distance, duration, cost = planner.optimize_route_greedy(start='Oslo', end='Barcelona', destinations=['Stockholm', 'Helsinki', 'Barcelona', 'Berlin', 'Vilnius', 'Tallinn', 'Copenhagen', 'Frankfurt', 'Warsaw', 'Rome'], criteria_weights=criteria_weights)
    end = time.time()

    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')
    print(f'Execution time: {end - start:.5f} seconds')

    print('\nOptimizing route from Oslo to Barcelona: simulated annealing')
    start = time.time()
    route, distance, duration, cost = planner.optimize_route_simulated_annealing(start='Oslo', end=None, destinations=['Stockholm', 'Helsinki', 'Barcelona', 'Berlin', 'Vilnius', 'Tallinn', 'Copenhagen', 'Frankfurt', 'Warsaw', 'Rome'], criteria_weights=criteria_weights)
    end = time.time()

    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')
    print(f'Execution time: {end - start:.5f} seconds')

    print('\nOptimizing route from Oslo to Barcelona: floyd warshall')
    start = time.time()
    route, distance, duration, cost = planner.optimize_route_floyd_warshall(start='Oslo', end=None, destinations=['Stockholm', 'Helsinki', 'Barcelona', 'Berlin', 'Vilnius', 'Tallinn', 'Copenhagen', 'Frankfurt', 'Warsaw', 'Rome'], criteria_weights=criteria_weights)
    end = time.time()

    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')
    print(f'Execution time: {end - start:.5f} seconds')

    print('\nOptimizing route from Oslo to Barcelona: dijkstra')
    start = time.time()
    route, distance, duration, cost = planner.optimize_route_dijkstra(start='Oslo', end=None, destinations=['Stockholm', 'Helsinki', 'Barcelona', 'Berlin', 'Vilnius', 'Tallinn', 'Copenhagen', 'Frankfurt', 'Warsaw', 'Rome'], criteria_weights=criteria_weights)
    end = time.time()

    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')
    print(f'Execution time: {end - start:.5f} seconds')

    print('\nOptimizing route from Oslo to Barcelona: A*')
    start = time.time()
    route, distance, duration, cost = planner.optimize_route_a_star(start='Oslo', end='Barcelona', destinations=['Stockholm', 'Helsinki', 'Barcelona', 'Berlin', 'Vilnius', 'Tallinn', 'Copenhagen', 'Frankfurt', 'Warsaw', 'Rome'])
    end = time.time()

    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')
    print(f'Execution time: {end - start:.5f} seconds')

if __name__ == '__main__':
    main()
