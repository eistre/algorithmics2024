from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from lib.planner import TravelPlanner, CriteriaWeight, Criteria
from lib.sample_loader import load_cities_with_coordinates

class CriteriaRequest(BaseModel):
    distance: float = 1.0
    duration: float = 0.0
    cost: float = 0.0

class PlanRequest(BaseModel):
    start: str
    destinations: list[str] = []
    algorithm: Literal['brute_force', 'greedy', 'simulated_annealing', 'floyd_warshall', 'dijkstra', 'a_star'] = 'greedy'
    criteria: CriteriaRequest = CriteriaRequest()

app = FastAPI()
CITY_LIST = load_cities_with_coordinates()
PLANNER = TravelPlanner(load_from_file=True)

@app.get('/cities')
def get_cities():
    return list(map(lambda city: city[0], CITY_LIST))

@app.get('/algorithms')
def get_algorithms():
    return PLANNER.algorithm_map.keys()

@app.post('/plan')
def plan_route(plan_request: PlanRequest):
    route, distance, duration, cost = PLANNER.optimize_route(
        start=plan_request.start,
        destinations=plan_request.destinations,
        algorithm=plan_request.algorithm,
        criteria_weights=[
            CriteriaWeight(Criteria.DISTANCE, plan_request.criteria.distance),
            CriteriaWeight(Criteria.DURATION, plan_request.criteria.duration),
            CriteriaWeight(Criteria.COST, plan_request.criteria.cost)
        ]
    )

    route_with_coordinates = [(city, next(city_data[1:] for city_data in CITY_LIST if city_data[0] == city)) for city in route]

    return {
        'route': route_with_coordinates,
        'distance': distance,
        'duration': duration,
        'cost': cost
    }
