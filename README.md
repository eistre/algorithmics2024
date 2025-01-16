# Algorithmics Group Project 2024

This repository contains the implementation of an efficient travel planning system using graph algorithms.

## Project Overview
A travel route optimization system designed to help travelers plan their journeys efficiently. The project focuses on finding optimal routes between multiple destinations using advanced graph algorithms, considering factors such as distance, time, and cost.

### Core Features
- Route optimization based on multiple criteria weights (distance, time, cost)
- Custom destination input support
- Flexible route planning with configurable start/end points
- Interactive map visualization of the optimized route

### Algorithms Used
- Brute Force
- Greedy (Nearest Neighbor)
- Floyd-Warshall
- Dijkstra
- A* (A Star)
- Simulated Annealing

### Materials Used
- Lecture slides & homeworks for the algorithms
- Claude AI
- ChatGPT
- OpenRouteService API (To calculate distance matrix)
- OpenStreetMap data
- FastAPI (For backend development)
- Maplibre GL JS (For map visualization)
- Vite & React (For frontend development)
- shadcn (For UI design)

## Getting Started

### Prerequisites
- Python 3.12 or higher
- Node.js 22 or higher
- REST Client for API testing (e.g. Postman, REST Client VSCode extension) (optional)
- Docker (optional)

### Installation
1. Clone the repository
    ```bash
    git clone https://github.com/eistre/algorithmics2024.git
    cd algorithmics2024
    ```

2. Create and activate virtual environment
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate
    ```

3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

4. To run the backend locally
    ```bash
    uvicorn main:app --reload
    ```

5. To run the frontend locally
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

6. To run the application with docker-compose
    ```bash
    docker-compose up --build
    ```

## Team Members

### Team Travel
- Taavi Eistre
- Marilin Ahvenainen
