export interface City {
  name: string;
  coordinates: [number, number]; // [latitude, longitude]
}

export interface CriteriaRequest {
  distance: number;
  duration: number;
  cost: number;
}

export interface PlanRequest {
  start: string;
  end: string | undefined;
  destinations: string[] | undefined;
  algorithm: string;
  criteria: CriteriaRequest | undefined;
}

export interface PlanResponse {
  route: string[];
  distance: number;
  duration: number;
  cost: number;
}
