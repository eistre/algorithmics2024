import { City, PlanRequest, PlanResponse } from '../types/types';
import axios from 'axios';

export async function getCities(): Promise<City[]> {
  const cities = await axios.get<City[]>('/api/cities', {
    headers: {
      'Content-Type': 'application/json'
    }
  })

  return cities.data;
}

export async function getAlgorithms(): Promise<string[]> {
  const algorithms = await axios.get<string[]>('/api/algorithms', {
    headers: {
      'Content-Type': 'application/json'
    }
  })

  return algorithms.data;
}

export async function optimizeRoute(planRequest: PlanRequest): Promise<PlanResponse> {
  const planResponse = await axios.post<PlanResponse>('/api/plan', planRequest, {
    headers: {
      'Content-Type': 'application/json'
    }
  })

  return planResponse.data;
}
