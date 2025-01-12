def load_cities_with_coordinates():
    """Loads the cities with their coordinates from a file."""
    with open('sample_data/cities_coordinates.txt') as f:
        cities = []
        for line in f:
            city, lat, lon = line.strip().split(',')
            cities.append((city, float(lat), float(lon)))
    return cities

def save_cities_with_coordinates(cities):
    """Saves the cities with their coordinates to a file."""
    with open('sample_data/cities_coordinates.txt', 'w') as f:
        for city, lat, lon in cities:
            f.write(f'{city},{lat},{lon}\n')

def load_distance_time_cost_matrix():
    """Loads the distance, time and cost matrix from a file."""
    with open('sample_data/matrix.txt') as f:
        matrix = []
        for line in f:
            row = []
            for cell in line.strip().split(','):
                distance, duration, cost = map(float, cell.split(';'))
                row.append((distance, duration, cost))
            matrix.append(row)
    return matrix

def save_distance_time_cost_matrix(matrix):
    """Saves the distance, time and cost matrix to a file."""
    with open('sample_data/matrix.txt', 'w') as f:
        for row in matrix:
            f.write(','.join([f'{distance};{duration};{cost}' for distance, duration, cost in row]) + '\n')