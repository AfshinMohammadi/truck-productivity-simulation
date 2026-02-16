"""
Basic Simulation Example
========================

Demonstrates how to run a simple simulation and analyze results.
"""

import sys
sys.path.insert(0, '..')

from src.simulation import Simulation, SimulationConfig
from src.environment import RoadNetwork, NetworkType
from src.utils import Visualizer


def main():
    """Run a basic simulation example."""
    
    print("=" * 60)
    print("Agent-Based Truck Productivity Simulation")
    print("=" * 60)
    print()
    
    # Create network
    print("Creating road network...")
    network = RoadNetwork(
        network_type=NetworkType.GRID,
        config={
            "network": {
                "dimensions": [10, 10],
                "edge_length": 1.0
            }
        }
    )
    network._create_grid_network({"dimensions": [10, 10], "edge_length": 1.0})
    network.depots = {"n_0_0", "n_9_9"}
    print(f"  Nodes: {network.graph.number_of_nodes()}")
    print(f"  Edges: {network.graph.number_of_edges()}")
    print()
    
    # Configure simulation
    config = SimulationConfig(
        n_trucks=50,
        duration=480,  # 8 hours
        routing_policy="congestion_aware",
        demand_rate=15,
        seed=42,
        verbose=True
    )
    
    # Create and run simulation
    print("Running simulation...")
    sim = Simulation(network=network, config=config)
    
    def progress_callback(progress):
        print(f"\r  Progress: {progress*100:.1f}%", end="", flush=True)
    
    metrics = sim.run(progress_callback=progress_callback)
    print("\n")
    
    # Display results
    print("Simulation Results:")
    print("-" * 40)
    print(f"  Total Deliveries: {metrics.total_deliveries}")
    print(f"  Failed Deliveries: {metrics.failed_deliveries}")
    print(f"  Total Distance: {metrics.total_distance:.2f} km")
    print(f"  Total Fuel: {metrics.total_fuel:.2f} L")
    print(f"  Average Speed: {metrics.avg_speed:.2f} km/h")
    print(f"  Average Trip Time: {metrics.avg_trip_time:.2f} min")
    print(f"  Fleet Efficiency: {metrics.fleet_efficiency:.4f} deliveries/km")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    viz = Visualizer(metrics)
    
    # Plot productivity timeline
    fig = viz.plot_productivity_timeline(
        metrics,
        title="Truck Fleet Productivity Analysis",
        save_path="../results/figures/productivity_timeline.png"
    )
    print("  Saved: productivity_timeline.png")
    
    # Plot network with congestion
    congestion = sim.traffic_model.get_congestion()
    fig = viz.plot_network(
        sim.network,
        congestion=congestion,
        title="Final Network Congestion State",
        save_path="../results/figures/network_congestion.png"
    )
    print("  Saved: network_congestion.png")
    
    print()
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
