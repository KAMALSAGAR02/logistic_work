"""
Meridian Retail Logistics (MRL) - Last-Mile Vehicle Route Optimization
Implements Clarke-Wright Savings Heuristic with 2-Opt local search
and compares against the baseline Dispatcher Sweep heuristic.
"""

import numpy as np
import pandas as pd

def generate_route_simulation(n_stops=40, seed=42, vehicle_capacity_kg=250.0):
    """
    Generates 40 customer delivery stops around a central distribution depot.
    Depot: (0, 0)
    Stops: (x, y) coordinates within [-25, 25] km grid.
    Demands: drawn from real parcel weights (mean ~9.4 kg).
    """
    rng = np.random.default_rng(seed)
    
    # Depot
    depot = np.array([0.0, 0.0])
    
    # 40 Stops coordinates (km)
    coords = rng.uniform(-25.0, 25.0, size=(n_stops, 2)).round(2)
    # Stop weights
    demands = np.round(rng.lognormal(mean=2.1, sigma=0.5, size=n_stops), 1)
    demands = np.clip(demands, 2.0, 31.5)
    
    # Save sample stops to data/route_stops_sample.csv
    stops_df = pd.DataFrame({
        "stop_id": [f"STOP-{i+1:02d}" for i in range(n_stops)],
        "x_km": coords[:, 0],
        "y_km": coords[:, 1],
        "demand_kg": demands
    })
    stops_df.to_csv("data/route_stops_sample.csv", index=False)
    
    # Full distance matrix (index 0 is depot, 1..n are stops)
    all_points = np.vstack([depot, coords])
    N = len(all_points)
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = np.linalg.norm(all_points[i] - all_points[j])
            
    return depot, coords, demands, dist_matrix, vehicle_capacity_kg

def route_distance(route, dist_matrix):
    """Calculate round-trip distance starting and ending at depot (node 0)."""
    if not route: return 0.0
    d = dist_matrix[0, route[0]]
    for i in range(len(route) - 1):
        d += dist_matrix[route[i], route[i+1]]
    d += dist_matrix[route[-1], 0]
    return d

def two_opt(route, dist_matrix):
    """Applies 2-opt local search to improve an existing route."""
    best = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if route_distance(new_route, dist_matrix) < route_distance(best, dist_matrix) - 1e-4:
                    best = new_route
                    improved = True
        route = best
    return best

def solve_clarke_wright(coords, demands, dist_matrix, capacity=250.0):
    """Clarke-Wright parallel savings heuristic with 2-opt post-optimization."""
    n = len(demands)
    # 1. Compute savings s(i, j) = d(0, i) + d(0, j) - d(i, j)
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Initialize each customer in their own route 0 -> i -> 0
    routes = {i: [i] for i in range(1, n + 1)}
    route_loads = {i: demands[i - 1] for i in range(1, n + 1)}
    
    for s, i, j in savings:
        # Find which route contains i and j
        r_i = None
        r_j = None
        for r_id, r in routes.items():
            if r_i is None and (r[0] == i or r[-1] == i):
                r_i = r_id
            if r_j is None and (r[0] == j or r[-1] == j):
                r_j = r_id
                
        if r_i is not None and r_j is not None and r_i != r_j:
            # Check capacity
            if route_loads[r_i] + route_loads[r_j] <= capacity:
                route_a = routes[r_i]
                route_b = routes[r_j]
                
                # Merge logic depending on which ends meet
                if route_a[-1] == i and route_b[0] == j:
                    merged = route_a + route_b
                elif route_a[0] == i and route_b[-1] == j:
                    merged = route_b + route_a
                elif route_a[-1] == i and route_b[-1] == j:
                    merged = route_a + route_b[::-1]
                elif route_a[0] == i and route_b[0] == j:
                    merged = route_a[::-1] + route_b
                else:
                    continue
                    
                del routes[r_i]
                del routes[r_j]
                routes[r_i] = merged
                route_loads[r_i] = route_loads[r_i] + route_loads[r_j]
                del route_loads[r_j]
                
    # 3. Apply 2-opt to each route
    optimized_routes = []
    total_km = 0.0
    for r in routes.values():
        opt_r = two_opt(r, dist_matrix)
        optimized_routes.append(opt_r)
        total_km += route_distance(opt_r, dist_matrix)
        
    return optimized_routes, round(total_km, 2)

def solve_dispatcher_sweep(coords, demands, dist_matrix, capacity=250.0):
    """Baseline dispatcher sweep algorithm (polar angle sorting)."""
    n = len(demands)
    # Compute polar angle from depot
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    order = np.argsort(angles) + 1 # 1-indexed for dist_matrix
    
    routes = []
    cur_route = []
    cur_load = 0.0
    
    for stop in order:
        demand = demands[stop - 1]
        if cur_load + demand <= capacity:
            cur_route.append(stop)
            cur_load += demand
        else:
            if cur_route: routes.append(cur_route)
            cur_route = [stop]
            cur_load = demand
    if cur_route: routes.append(cur_route)
    
    total_km = sum(route_distance(r, dist_matrix) for r in routes)
    return routes, round(total_km, 2)

def run_route_comparison(seed=42):
    depot, coords, demands, dist_matrix, capacity = generate_route_simulation(n_stops=40, seed=seed)
    
    sweep_routes, sweep_km = solve_dispatcher_sweep(coords, demands, dist_matrix, capacity)
    cw_routes, cw_km = solve_clarke_wright(coords, demands, dist_matrix, capacity)
    
    savings_pct = round(((sweep_km - cw_km) / sweep_km) * 100, 1)
    cost_per_km = 1.20 # $1.20 per vehicle km
    dollar_saving = round((sweep_km - cw_km) * cost_per_km, 2)
    
    summary = {
        "Stops": len(coords),
        "Dispatcher Sweep Km": sweep_km,
        "Dispatcher Vans": len(sweep_routes),
        "Clarke-Wright Km": cw_km,
        "Clarke-Wright Vans": len(cw_routes),
        "Distance Reduction (%)": savings_pct,
        "Daily Dollar Savings ($)": dollar_saving,
        "Annual Savings (250 days) ($)": round(dollar_saving * 250, 2)
    }
    return summary, sweep_routes, cw_routes, depot, coords

if __name__ == "__main__":
    summary, sweep_r, cw_r, depot, coords = run_route_comparison()
    print("Route Optimization Comparison:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
