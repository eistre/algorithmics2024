import asyncio
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from lib.planner import TravelPlanner, CriteriaWeight, Criteria

class CriteriaRequest(BaseModel):
    distance: float = 1.0
    duration: float = 0.0
    cost: float = 0.0

class PlanRequest(BaseModel):
    start: str
    end: str | None = None
    destinations: list[str] = []
    algorithm: Literal['brute_force', 'greedy', 'simulated_annealing', 'floyd_warshall', 'dijkstra', 'a_star'] = 'greedy'
    criteria: CriteriaRequest = CriteriaRequest()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

PLANNER = TravelPlanner(load_from_file=True)

@app.get('/cities')
def get_cities():
    return list(map(lambda city: {'name': city[0], 'coordinates': city[1:]}, PLANNER.coordinates))

@app.get('/algorithms')
def get_algorithms():
    return list(PLANNER.algorithm_map.keys())

@app.post('/plan')
async def plan_route(plan_request: PlanRequest):
    try:
        route, distance, duration, cost = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, lambda: PLANNER.optimize_route(
                start=plan_request.start,
                end=plan_request.end,
                destinations=plan_request.destinations,
                algorithm=plan_request.algorithm,
                criteria_weights=[
                    CriteriaWeight(Criteria.DISTANCE, plan_request.criteria.distance),
                    CriteriaWeight(Criteria.DURATION, plan_request.criteria.duration),
                    CriteriaWeight(Criteria.COST, plan_request.criteria.cost)
                ]
            )),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail='Request timed out after 60 seconds')

    return {
        'route': route,
        'distance': distance,
        'duration': duration,
        'cost': cost
    }
