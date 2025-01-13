from planner import TravelPlanner

def main():
    planner = TravelPlanner(load_from_file=True)

    print('\nOptimizing route from Oslo to Barcelona: brute force')
    route, distance, duration, cost = planner.optimize_route_brute_force(start='Oslo', end='Barcelona', destinations=['Stockholm', 'Helsinki', 'Barcelona', 'Berlin', 'Vilnius', 'Tallinn', 'Copenhagen'])
    
    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')

    print('\nOptimizing route from Oslo to Barcelona: greedy')
    route, distance, duration, cost = planner.optimize_route_greedy(start='Oslo', end='Barcelona', destinations=['Stockholm', 'Barcelona', 'Berlin', 'Helsinki', 'Vilnius', 'Tallinn', 'Copenhagen'])
    
    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')

    print('\nOptimizing route from Oslo to Barcelona: simulated annealing')
    route, distance, duration, cost = planner.optimize_route_simulated_annealing(start='Oslo', end='Barcelona', destinations=['Stockholm', 'Helsinki', 'Barcelona', 'Berlin', 'Vilnius', 'Tallinn', 'Copenhagen'])
    
    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')

if __name__ == '__main__':
    main()
