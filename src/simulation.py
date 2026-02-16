"""
Simulation Orchestrator
=======================

Main simulation controller that coordinates agents, environment,
and data collection for agent-based logistics simulations.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import random
from collections import defaultdict

import numpy as np

from src.agents.truck_agent import TruckAgent, TruckType, RoutingPolicy, DeliveryTask
from src.agents.traffic_agent import TrafficAgent
from src.environment.road_network import RoadNetwork, NetworkType
from src.environment.traffic_model import TrafficModel


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    n_trucks: int = 50
    truck_type_distribution: Dict[str, float] = field(default_factory=lambda: {"medium": 1.0})
    routing_policy: str = "congestion_aware"
    duration: float = 480.0  # minutes (8 hours)
    time_step: float = 1.0  # minutes
    demand_rate: float = 10.0  # tasks per hour
    seed: Optional[int] = None
    verbose: bool = False


@dataclass
class SimulationMetrics:
    """Collected metrics from simulation."""
    total_deliveries: int = 0
    failed_deliveries: int = 0
    total_distance: float = 0.0
    total_fuel: float = 0.0
    total_waiting_time: float = 0.0
    avg_trip_time: float = 0.0
    avg_speed: float = 0.0
    network_utilization: float = 0.0
    fleet_efficiency: float = 0.0
    
    # Time series data
    deliveries_over_time: List[int] = field(default_factory=list)
    avg_speed_over_time: List[float] = field(default_factory=list)
    congestion_over_time: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_deliveries": self.total_deliveries,
            "failed_deliveries": self.failed_deliveries,
            "total_distance": self.total_distance,
            "total_fuel": self.total_fuel,
            "total_waiting_time": self.total_waiting_time,
            "avg_trip_time": self.avg_trip_time,
            "avg_speed": self.avg_speed,
            "network_utilization": self.network_utilization,
            "fleet_efficiency": self.fleet_efficiency,
        }


class Simulation:
    """
    Main simulation controller.
    
    Orchestrates the agent-based simulation by:
    - Initializing network and traffic model
    - Creating and managing agents
    - Running the simulation loop
    - Collecting and aggregating metrics
    
    Example:
        >>> config = SimulationConfig(n_trucks=50, duration=480)
        >>> sim = Simulation(config=config)
        >>> results = sim.run()
        >>> print(results.total_deliveries)
    """
    
    def __init__(
        self,
        network: Optional[RoadNetwork] = None,
        n_trucks: int = 50,
        duration: float = 480.0,
        config: Optional[SimulationConfig] = None,
        **kwargs
    ):
        """
        Initialize simulation.
        
        Args:
            network: Road network (created if None)
            n_trucks: Number of trucks (overridden by config)
            duration: Simulation duration in minutes (overridden by config)
            config: Full simulation configuration
            **kwargs: Additional configuration parameters
        """
        # Use config if provided, otherwise create from kwargs
        if config is None:
            config = SimulationConfig(
                n_trucks=n_trucks,
                duration=duration,
                **kwargs
            )
        
        self.config = config
        
        # Set random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        # Initialize network
        self.network = network or self._create_default_network()
        
        # Initialize traffic model
        self.traffic_model = TrafficModel(self.network)
        
        # Initialize agents
        self.trucks: List[TruckAgent] = []
        self.traffic_agents: List[TrafficAgent] = []
        self._create_agents()
        
        # Task management
        self.pending_tasks: List[DeliveryTask] = []
        self.completed_tasks: List[DeliveryTask] = []
        self.task_id_counter: int = 0
        
        # Simulation state
        self.current_time: float = 0.0
        self.is_running: bool = False
        
        # Metrics collection
        self.metrics = SimulationMetrics()
        self._trip_times: List[float] = []
        
    def _create_default_network(self) -> RoadNetwork:
        """Create a default grid network."""
        network = RoadNetwork(
            network_type=NetworkType.GRID,
            config={
                "network": {
                    "type": "grid",
                    "dimensions": [8, 8],
                    "edge_length": 1.0,
                },
                "edges": {
                    "free_flow_speed": 40.0,
                    "capacity": 50.0,
                    "lanes": 1,
                }
            }
        )
        network._create_grid_network({"dimensions": [8, 8], "edge_length": 1.0})
        network.depots = {"n_0_0", "n_7_7"}
        return network
    
    def _create_agents(self) -> None:
        """Create truck agents based on configuration."""
        truck_types = {
            "small": TruckType.SMALL,
            "medium": TruckType.MEDIUM,
            "large": TruckType.LARGE,
        }
        
        routing_policies = {
            "shortest": RoutingPolicy.SHORTEST,
            "fastest": RoutingPolicy.FASTEST,
            "congestion_aware": RoutingPolicy.CONGESTION_AWARE,
            "adaptive": RoutingPolicy.ADAPTIVE,
        }
        
        policy = routing_policies.get(self.config.routing_policy, RoutingPolicy.CONGESTION_AWARE)
        
        depots = list(self.network.depots)
        if not depots:
            depots = [list(self.network.graph.nodes())[0]]
        
        for i in range(self.config.n_trucks):
            # Determine truck type from distribution
            type_roll = random.random()
            cumulative = 0.0
            
            selected_type = TruckType.MEDIUM
            for type_name, prob in self.config.truck_type_distribution.items():
                cumulative += prob
                if type_roll <= cumulative:
                    selected_type = truck_types.get(type_name, TruckType.MEDIUM)
                    break
            
            # Assign to random depot
            depot = random.choice(depots)
            
            truck = TruckAgent(
                agent_id=f"truck_{i:03d}",
                name=f"Truck_{i}",
                truck_type=selected_type,
                routing_policy=policy,
                start_node=depot,
                depot_node=depot,
                verbose=self.config.verbose
            )
            
            self.trucks.append(truck)
    
    def generate_demand(self) -> None:
        """Generate delivery tasks based on demand rate."""
        # Expected tasks for this time step
        expected_tasks = self.config.demand_rate * (self.config.time_step / 60.0)
        n_new_tasks = np.random.poisson(expected_tasks)
        
        depots = list(self.network.depots)
        delivery_points = list(self.network.delivery_points)
        
        if not delivery_points:
            # Use random nodes as delivery points
            all_nodes = list(set(self.network.graph.nodes()) - set(depots))
            if all_nodes:
                delivery_points = random.sample(all_nodes, min(10, len(all_nodes)))
            else:
                delivery_points = list(self.network.graph.nodes())
        
        for _ in range(n_new_tasks):
            origin = random.choice(depots)
            destination = random.choice(delivery_points)
            
            if origin == destination:
                continue
            
            task = DeliveryTask(
                task_id=f"task_{self.task_id_counter:05d}",
                origin=origin,
                destination=destination,
                demand=random.uniform(0.5, 2.0),
                priority=random.randint(1, 3)
            )
            
            self.pending_tasks.append(task)
            self.task_id_counter += 1
    
    def assign_tasks(self) -> None:
        """Assign pending tasks to available trucks."""
        # Sort tasks by priority (higher first)
        self.pending_tasks.sort(key=lambda t: -t.priority)
        
        # Get available trucks
        available_trucks = [
            t for t in self.trucks
            if t.state.name == "IDLE" and t.truck_state.current_task is None
        ]
        
        for task in self.pending_tasks[:]:
            if not available_trucks:
                break
            
            # Find best truck for this task (simple: closest)
            best_truck = None
            best_distance = float('inf')
            
            for truck in available_trucks:
                if truck.truck_state.current_node:
                    try:
                        path = self.network.shortest_path(
                            truck.truck_state.current_node,
                            task.origin
                        )
                        if path:
                            # Estimate distance
                            dist = len(path)  # Simplified
                            if dist < best_distance:
                                best_distance = dist
                                best_truck = truck
                    except:
                        continue
            
            if best_truck:
                if best_truck.assign_task(task):
                    self.pending_tasks.remove(task)
                    available_trucks.remove(best_truck)
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Returns:
            Dictionary with step summary
        """
        dt = self.config.time_step
        
        # Generate demand
        self.generate_demand()
        
        # Assign tasks
        self.assign_tasks()
        
        # Create environment context
        environment = type('Environment', (), {
            'network': self.network,
            'traffic_model': self.traffic_model,
            'current_time': self.current_time
        })()
        
        # Update traffic
        active_vehicles = [t for t in self.trucks if t.state.name == "TRAVELING"]
        self.traffic_model.update(active_vehicles, dt)
        
        # Step each truck
        for truck in self.trucks:
            result = truck.step(environment, dt)
            
            # Track completed deliveries
            if result.get("status") == "delivered":
                self.metrics.total_deliveries += 1
                if truck.truck_state.current_task:
                    self._trip_times.append(
                        self.current_time - truck.truck_state.current_task.time_window[0]
                        if truck.truck_state.current_task.time_window
                        else self.current_time
                    )
        
        # Update time
        self.current_time += dt
        self.network.update_time(dt)
        
        # Collect time series data
        self._collect_time_series()
        
        return {
            "time": self.current_time,
            "active_trucks": len(active_vehicles),
            "pending_tasks": len(self.pending_tasks),
            "deliveries": self.metrics.total_deliveries
        }
    
    def _collect_time_series(self) -> None:
        """Collect time series metrics."""
        self.metrics.deliveries_over_time.append(self.metrics.total_deliveries)
        
        speeds = [t.truck_state.speed for t in self.trucks if t.truck_state.speed > 0]
        self.metrics.avg_speed_over_time.append(
            sum(speeds) / len(speeds) if speeds else 0
        )
        
        congestion = self.traffic_model.get_network_congestion_summary()
        self.metrics.congestion_over_time.append(congestion["avg_congestion"])
    
    def run(
        self,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> SimulationMetrics:
        """
        Run the complete simulation.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            SimulationMetrics with final results
        """
        self.is_running = True
        n_steps = int(self.config.duration / self.config.time_step)
        
        start_time = time.time()
        
        for step in range(n_steps):
            self.step()
            
            if progress_callback and step % 10 == 0:
                progress = self.current_time / self.config.duration
                progress_callback(progress)
        
        # Finalize metrics
        self._finalize_metrics()
        
        elapsed = time.time() - start_time
        if self.config.verbose:
            print(f"Simulation completed in {elapsed:.2f}s")
        
        self.is_running = False
        return self.metrics
    
    def _finalize_metrics(self) -> None:
        """Calculate final aggregate metrics."""
        # Aggregate truck metrics
        for truck in self.trucks:
            truck.finalize_metrics()
            
            self.metrics.total_distance += truck.metrics.total_distance
            self.metrics.total_fuel += truck.metrics.fuel_consumed
            self.metrics.total_waiting_time += truck.metrics.waiting_time
            self.metrics.failed_deliveries += truck.metrics.deliveries_failed
        
        # Calculate averages
        if self._trip_times:
            self.metrics.avg_trip_time = sum(self._trip_times) / len(self._trip_times)
        
        speeds = [t.metrics.avg_speed for t in self.trucks if t.metrics.avg_speed > 0]
        if speeds:
            self.metrics.avg_speed = sum(speeds) / len(speeds)
        
        # Network utilization
        total_capacity = sum(
            t.truck_state.cargo_capacity for t in self.trucks
        )
        total_load = sum(
            t.truck_state.cargo_load for t in self.trucks
        )
        self.metrics.network_utilization = total_load / total_capacity if total_capacity > 0 else 0
        
        # Fleet efficiency (deliveries per distance)
        if self.metrics.total_distance > 0:
            self.metrics.fleet_efficiency = (
                self.metrics.total_deliveries / self.metrics.total_distance
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            "time": self.current_time,
            "trucks": [t.get_state_summary() for t in self.trucks],
            "traffic": self.traffic_model.get_network_congestion_summary(),
            "pending_tasks": len(self.pending_tasks),
            "deliveries": self.metrics.total_deliveries,
        }
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.current_time = 0.0
        self.metrics = SimulationMetrics()
        self.pending_tasks.clear()
        self.completed_tasks.clear()
        self.task_id_counter = 0
        self._trip_times.clear()
        
        for truck in self.trucks:
            truck.reset()
        
        self.traffic_model.reset()
        self.network.clear_cache()
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'Simulation':
        """Create simulation from configuration file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = SimulationConfig(**config_dict.get('simulation', {}))
        
        network = RoadNetwork.from_config(config_path) if 'network' in config_dict else None
        
        return cls(network=network, config=config)
