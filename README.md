# Agent-Based Modeling for Truck Productivity under Traffic Constraints

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A sophisticated agent-based simulation framework for optimizing logistics operations under traffic congestion constraints. This project implements multi-agent systems, constraint-based optimization, and high-dimensional parameter space exploration for analyzing truck productivity in complex transportation networks.

## Overview

This simulation framework models truck fleet operations where individual agents (trucks) navigate a road network subject to traffic constraints, congestion dynamics, and operational objectives. The system enables analysis of:

- **System-level throughput** under varying congestion scenarios
- **Fleet efficiency** across different routing strategies
- **Network stability** under high-demand conditions
- **Optimal fleet configurations** for given operational constraints

## Key Features

- **Agent-Based Architecture**: Modular truck agents with customizable behavior policies
- **Dynamic Traffic Modeling**: Realistic congestion propagation and dissipation
- **Multi-Objective Optimization**: Balance throughput, efficiency, and stability
- **Parameter Sweep Framework**: Systematic exploration of high-dimensional design spaces
- **Visualization Tools**: Comprehensive analysis and result visualization
- **Extensible Design**: Easy integration of new agent types and network topologies

## Installation

```bash
# Clone the repository
git clone https://github.com/afshinmohammadi/truck-productivity-simulation.git
cd truck-productivity-simulation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from src.simulation import Simulation
from src.agents import TruckAgent
from src.environment import RoadNetwork

# Create road network
network = RoadNetwork.from_config('configs/network.yaml')

# Initialize simulation
sim = Simulation(
    network=network,
    n_trucks=50,
    duration=480,  # 8-hour shift in minutes
    seed=42
)

# Run simulation
results = sim.run()

# Analyze results
results.plot_productivity()
results.summary_statistics()
```

## Project Structure

```
truck-productivity-simulation/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py      # Abstract base class for agents
│   │   ├── truck_agent.py     # Truck agent implementation
│   │   └── traffic_agent.py   # Traffic management agent
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── road_network.py    # Network topology and dynamics
│   │   ├── traffic_model.py   # Congestion modeling
│   │   └── demand_generator.py# Demand pattern generation
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── parameter_sweep.py # High-dimensional sweep framework
│   │   ├── objective.py       # Multi-objective functions
│   │   └── optimizer.py       # Optimization algorithms
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py   # Plotting and visualization
│       └── metrics.py         # Performance metrics
├── notebooks/
│   ├── 01_basic_simulation.ipynb
│   ├── 02_parameter_sweep.ipynb
│   └── 03_optimization.ipynb
├── configs/
│   ├── network.yaml           # Network configuration
│   ├── agents.yaml            # Agent parameters
│   └── sweep.yaml             # Parameter sweep config
├── results/                   # Output directory
├── docs/                      # Documentation
├── requirements.txt
├── setup.py
└── README.md
```

## Simulation Components

### Agent Types

| Agent Type | Description | Key Parameters |
|------------|-------------|----------------|
| `TruckAgent` | Individual truck with routing decisions | speed, capacity, routing_policy |
| `TrafficAgent` | Network-level traffic management | signal_timing, flow_control |

### Network Model

The road network is modeled as a directed graph with:
- **Nodes**: Intersections, depots, delivery points
- **Edges**: Road segments with capacity, speed limit, length
- **Dynamic attributes**: Current flow, congestion level

### Traffic Dynamics

Congestion is modeled using a modified Cell Transmission Model (CTM):
- Flow propagation based on density
- Shockwave formation and dissipation
- Queue spillback effects

## Parameter Sweep Framework

Explore high-dimensional design spaces systematically:

```python
from src.optimization import ParameterSweep

# Define parameter ranges
param_ranges = {
    'n_trucks': [20, 30, 40, 50, 60],
    'routing_policy': ['shortest', 'congestion_aware', 'adaptive'],
    'departure_strategy': ['uniform', 'staggered', 'demand_driven'],
    'fleet_mix': [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5)],  # (small, large)
    'traffic_signal_timing': [60, 90, 120]
}

# Run sweep
sweep = ParameterSweep(param_ranges, n_replicates=10)
results = sweep.run_parallel(n_workers=8)

# Analyze
results.plot_pareto_front()
results.sensitivity_analysis()
```

## Optimization Objectives

The framework supports multi-objective optimization:

1. **Throughput**: Total deliveries completed per time unit
2. **Efficiency**: Distance traveled per delivery
3. **Stability**: Variance in system performance
4. **Cost**: Operational costs including fuel and time

```python
from src.optimization import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(
    objectives=['throughput', 'efficiency', 'stability'],
    weights=[0.4, 0.3, 0.3]
)

optimal_config = optimizer.optimize(
    search_space=param_ranges,
    method='genetic_algorithm',
    max_generations=50
)
```

## Results Visualization

```python
from src.utils import Visualizer

viz = Visualizer(results)

# Productivity over time
viz.plot_productivity_timeline()

# Network congestion heatmap
viz.plot_congestion_heatmap()

# Agent trajectories
viz.plot_agent_paths()

# Performance distributions
viz.plot_performance_boxplots()

# Sensitivity analysis
viz.plot_sensitivity_tornado()
```

## Example Results

Default configuration (50 trucks, 8-hour simulation):

| Metric | Mean | Std Dev |
|--------|------|---------|
| Total Deliveries | 342 | 18 |
| Avg Trip Time (min) | 47.3 | 12.1 |
| Avg Speed (km/h) | 28.6 | 4.2 |
| Network Utilization | 0.73 | 0.05 |
| Fleet Efficiency | 0.81 | 0.03 |

## Configuration

Network and agent parameters can be configured via YAML files:

```yaml
# configs/network.yaml
network:
  type: grid
  dimensions: [10, 10]
  edge_length: 1.0  # km

nodes:
  depot:
    count: 2
    placement: corners
  delivery:
    count: 20
    placement: random

edges:
  capacity: 50  # vehicles/hour
  free_flow_speed: 40  # km/h
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{truck_productivity_sim,
  author = {Mohammadi, Afshin},
  title = {Agent-Based Modeling for Truck Productivity under Traffic Constraints},
  year = {2024},
  url = {https://github.com/afshinmohammadi/truck-productivity-simulation}
}
```

## Contact

**Afshin Mohammadi**  
Email: Afshinciv@gmail.com  
