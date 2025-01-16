import { City, PlanRequest, PlanResponse } from '../types/types';
import axios from 'axios';

export async function getCities(): Promise<City[]> {
  const cities = await axios.get<City[]>('http://localhost:8000/cities', {
    headers: {
      'Content-Type': 'application/json'
    }
  })

  return cities.data;
}

export async function getAlgorithms(): Promise<string[]> {
  const algorithms = await axios.get<string[]>('http://localhost:8000/algorithms', {
    headers: {
      'Content-Type': 'application/json'
    }
  })

  return algorithms.data;
}

export async function optimizeRoute(planRequest: PlanRequest): Promise<PlanResponse> {
  const planResponse = await axios.post<PlanResponse>('http://localhost:8000/plan', planRequest, {
    headers: {
      'Content-Type': 'application/json'
    }
  })

  return planResponse.data;
}
