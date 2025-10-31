#!/usr/bin/env python3
"""
Debug why vehicles are stuck
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.traffic_model import TrafficSimulationModel

model = TrafficSimulationModel(max_vehicles=3, custom_lanes_path='eda/data/lanes.json')

print("Initial state:")
for v in model.vehicles:
    print(f"  Vehicle {v.unique_id}: lane={v.lane_id}, pos={v.position:.2f}m, speed={v.speed:.2f} m/s")
    leader = v.get_leader()
    if leader:
        dist = v.calculate_distance_to_leader()
        print(f"    Leader: Vehicle {leader.unique_id}, distance={dist:.2f}m")
    else:
        print(f"    No leader")

print("\nRunning 10 steps:")
for i in range(10):
    model.step()
    if i % 2 == 0:
        print(f"\nStep {i+1}:")
        for v in model.vehicles:
            print(f"  Vehicle {v.unique_id}: pos={v.position:.2f}m, speed={v.speed:.2f} m/s")
            leader = v.get_leader()
            if leader:
                dist = v.calculate_distance_to_leader()
                print(f"    Leader: Vehicle {leader.unique_id}, distance={dist:.2f}m")

