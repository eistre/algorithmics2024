from planner import TravelPlanner

def main():
    planner = TravelPlanner(load_from_file=True)

    print('Optimizing route from Oslo to Tallinn: greedy')
    route, distance, duration, cost = planner.optimize_route_greedy(start='Oslo', end='Tallinn', destinations=['Stockholm', 'Helsinki', 'Vilnius'])
    
    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')

    print('\nOptimizing route from Oslo to Tallinn: simulated annealing')
    route, distance, duration, cost = planner.optimize_route_simulated_annealing(start='Oslo', end='Tallinn', destinations=['Stockholm', 'Helsinki', 'Vilnius'])
    
    print('Route:', ' -> '.join(route))
    print(f'Distance: {distance:.2f} km')
    print(f'Duration: {duration:.2f} hours')
    print(f'Cost: {cost:.2f} €')

if __name__ == '__main__':
    main()
