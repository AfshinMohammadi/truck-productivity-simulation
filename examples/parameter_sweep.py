"""
Parameter Sweep Example
=======================

Demonstrates high-dimensional parameter space exploration
for analyzing simulation sensitivity and finding optimal configurations.
"""

import sys
sys.path.insert(0, '..')

from src.simulation import Simulation, SimulationConfig
from src.environment import RoadNetwork, NetworkType
from src.optimization import ParameterSweep, sensitivity_analysis
from src.utils import Visualizer
import time


def simulation_function(config):
    """
    Wrapper function for parameter sweep.
    
    Takes a configuration dictionary and returns metrics.
    """
    # Create network
    network = RoadNetwork(network_type=NetworkType.GRID)
    network._create_grid_network({"dimensions": [8, 8], "edge_length": 1.0})
    network.depots = {"n_0_0", "n_7_7"}
    
    # Extract parameters from config
    n_trucks = config.get('n_trucks', 50)
    routing_policy = config.get('routing_policy', 'congestion_aware')
    
    # Create simulation config
    sim_config = SimulationConfig(
        n_trucks=n_trucks,
        duration=240,  # 4 hours for faster sweep
        routing_policy=routing_policy,
        demand_rate=config.get('demand_rate', 10),
        seed=42
    )
    
    # Run simulation
    sim = Simulation(network=network, config=sim_config)
    metrics = sim.run()
    
    # Return metrics as dictionary
    return {
        "throughput": metrics.total_deliveries,
        "efficiency": metrics.fleet_efficiency,
        "avg_speed": metrics.avg_speed,
        "total_distance": metrics.total_distance,
        "avg_trip_time": metrics.avg_trip_time,
    }


def main():
    """Run parameter sweep example."""
    
    print("=" * 60)
    print("Parameter Sweep Analysis")
    print("=" * 60)
    print()
    
    # Define parameter ranges
    param_ranges = {
        'n_trucks': [20, 30, 40, 50, 60],
        'routing_policy': ['shortest', 'congestion_aware', 'adaptive'],
        'demand_rate': [5, 10, 15, 20],
    }
    
    # Create parameter sweep
    sweep = ParameterSweep(
        parameter_ranges=param_ranges,
        sampling_method="full_factorial",
        n_replicates=3,  # Reduced for demo
        parallel=True,
        n_workers=4,
        seed=42
    )
    
    # Estimate runtime
    print("Estimating runtime...")
    estimate = sweep.estimate_runtime(sample_runs=3, simulation_func=simulation_function)
    print(f"  Total configurations: {estimate['total_configs']}")
    print(f"  Total runs: {estimate['total_runs']}")
    print(f"  Estimated time: {estimate['estimated_parallel_time']:.1f}s (parallel)")
    print()
    
    # Run sweep
    print("Running parameter sweep...")
    start_time = time.time()
    
    def progress_callback(completed, total):
        pct = completed / total * 100
        print(f"\r  Progress: {completed}/{total} ({pct:.1f}%)", end="", flush=True)
    
    results = sweep.run(simulation_function, progress_callback)
    elapsed = time.time() - start_time
    print(f"\n\n  Completed in {elapsed:.2f}s")
    print()
    
    # Analyze results
    print("Analysis Results:")
    print("-" * 40)
    
    # Aggregate results
    aggregated = results.aggregate_by_config()
    print(f"  Configurations tested: {len(aggregated)}")
    
    # Find best configuration
    best_config = max(aggregated.values(), key=lambda x: x.get('throughput_mean', 0))
    print(f"\n  Best configuration (by throughput):")
    print(f"    Parameters: {best_config['parameters']}")
    print(f"    Throughput: {best_config['throughput_mean']:.1f} ± {best_config['throughput_std']:.1f}")
    print(f"    Efficiency: {best_config['efficiency_mean']:.4f}")
    
    # Sensitivity analysis
    print("\n  Sensitivity Analysis:")
    sensitivities = sensitivity_analysis(results, 'throughput')
    for param, sens in sensitivities.items():
        print(f"    {param}: {sens:.3f}")
    
    # Save results
    results.save('../results/sweep_results.json')
    print("\n  Results saved to: results/sweep_results.json")
    
    # Visualizations
    print("\n  Creating visualizations...")
    viz = Visualizer(results)
    
    # Performance boxplots
    viz.plot_performance_boxplots(
        results,
        metrics=['throughput', 'efficiency'],
        group_by='routing_policy',
        save_path="../results/figures/sweep_boxplots.png"
    )
    
    # Sensitivity tornado chart
    viz.plot_sensitivity_tornado(
        sensitivities,
        save_path="../results/figures/sensitivity_tornado.png"
    )
    
    # Pareto front
    pareto = results.get_pareto_front(
        objectives=['throughput', 'efficiency'],
        directions=['maximize', 'maximize']
    )
    
    if pareto:
        viz.plot_pareto_front(
            pareto,
            obj1='throughput',
            obj2='efficiency',
            save_path="../results/figures/pareto_front.png"
        )
    
    print("  Visualizations saved to: results/figures/")
    
    print()
    print("Parameter sweep completed successfully!")


if __name__ == "__main__":
    main()
