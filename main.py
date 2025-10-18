#!/usr/bin/env python3
"""
Main Traffic Simulation Script

This script runs the traffic simulation with PyGame visualization.
It demonstrates the Krauss car-following model and lane-changing behavior.
"""

import sys
import os
import argparse
from typing import Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.traffic_model import TrafficSimulationModel
from src.visualization.pygame_viz import TrafficVisualization


def main():
    """Main function to run the traffic simulation."""
    parser = argparse.ArgumentParser(description='Traffic Simulation with Krauss Model')
    
    # Simulation parameters
    parser.add_argument('--width', type=int, default=1000,
                      help='Simulation width (default: 1000)')
    parser.add_argument('--height', type=int, default=1000,
                      help='Simulation height (default: 1000)')
    parser.add_argument('--time-step', type=float, default=0.1,
                      help='Time step in seconds (default: 0.1)')
    parser.add_argument('--spawn-rate', type=float, default=0.1,
                      help='Vehicle spawn rate per second (default: 0.1)')
    parser.add_argument('--max-vehicles', type=int, default=50,
                      help='Maximum number of vehicles (default: 50)')
    
    # Visualization parameters
    parser.add_argument('--viz-width', type=int, default=1200,
                      help='Visualization window width (default: 1200)')
    parser.add_argument('--viz-height', type=int, default=800,
                      help='Visualization window height (default: 800)')
    parser.add_argument('--fps', type=int, default=60,
                      help='Target FPS (default: 60)')
    
    # Other options
    parser.add_argument('--headless', action='store_true',
                      help='Run without visualization (headless mode)')
    parser.add_argument('--steps', type=int, default=None,
                      help='Number of simulation steps to run (headless mode only)')
    
    args = parser.parse_args()
    
    print("Traffic Simulation with Krauss Car-Following Model")
    print("=" * 50)
    print(f"Simulation size: {args.width}x{args.height}")
    print(f"Time step: {args.time_step}s")
    print(f"Spawn rate: {args.spawn_rate} vehicles/s")
    print(f"Max vehicles: {args.max_vehicles}")
    print()
    
    # Create simulation model
    model = TrafficSimulationModel(
        width=args.width,
        height=args.height,
        time_step=args.time_step,
        vehicle_spawn_rate=args.spawn_rate,
        max_vehicles=args.max_vehicles
    )
    
    if args.headless:
        # Run headless simulation
        run_headless_simulation(model, args.steps)
    else:
        # Run with visualization
        run_visualized_simulation(model, args)
    
    print("Simulation completed!")


def run_headless_simulation(model: TrafficSimulationModel, steps: Optional[int]):
    """
    Run simulation without visualization.
    
    Args:
        model: Traffic simulation model
        steps: Number of steps to run (None for infinite)
    """
    print("Running headless simulation...")
    
    step_count = 0
    try:
        while True:
            model.step()
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                info = model.get_model_info()
                print(f"Step {step_count}: {info['num_vehicles']} vehicles, "
                      f"avg speed: {info['stats']['average_speed']:.1f} m/s")
            
            # Check if we should stop
            if steps is not None and step_count >= steps:
                break
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    # Print final statistics
    final_info = model.get_model_info()
    print("\nFinal Statistics:")
    print(f"Total steps: {step_count}")
    print(f"Final vehicles: {final_info['num_vehicles']}")
    print(f"Average speed: {final_info['stats']['average_speed']:.1f} m/s")
    print(f"Total spawned: {final_info['stats']['total_vehicles_spawned']}")
    print(f"Total removed: {final_info['stats']['total_vehicles_removed']}")


def run_visualized_simulation(model: TrafficSimulationModel, args):
    """
    Run simulation with PyGame visualization.
    
    Args:
        model: Traffic simulation model
        args: Command line arguments
    """
    print("Starting visualization...")
    
    # Create visualization
    visualization = TrafficVisualization(
        model=model,
        width=args.viz_width,
        height=args.viz_height,
        fps=args.fps
    )
    
    # Run visualization
    try:
        visualization.run()
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
