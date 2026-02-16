"""
Truck Agent Module
==================

Implements truck agents that navigate the road network, make routing
decisions, and complete delivery tasks. Supports multiple routing policies
and behavior configurations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import random
import math

from src.agents.base_agent import BaseAgent, AgentState, AgentMetrics


class RoutingPolicy(Enum):
    """Available routing policies for trucks."""
    SHORTEST = "shortest"  # Shortest path by distance
    FASTEST = "fastest"    # Fastest path considering current traffic
    CONGESTION_AWARE = "congestion_aware"  # Avoids congested roads
    ADAPTIVE = "adaptive"  # Dynamically adapts based on real-time conditions


class TruckType(Enum):
    """Types of trucks with different characteristics."""
    SMALL = "small"   # Light truck, faster, smaller capacity
    MEDIUM = "medium"  # Standard truck
    LARGE = "large"   # Heavy truck, slower, larger capacity


@dataclass
class TruckMetrics(AgentMetrics):
    """Extended metrics for truck agents."""
    deliveries_completed: int = 0
    deliveries_failed: int = 0
    avg_speed: float = 0.0
    congestion_encountered: float = 0.0  # Total time spent in congestion
    route_efficiency: float = 0.0  # Actual vs planned distance
    
    def to_dict(self) -> Dict[str, float]:
        base = super().to_dict()
        base.update({
            "deliveries_completed": self.deliveries_completed,
            "deliveries_failed": self.deliveries_failed,
            "avg_speed": self.avg_speed,
            "congestion_encountered": self.congestion_encountered,
            "route_efficiency": self.route_efficiency,
        })
        return base


@dataclass
class DeliveryTask:
    """Represents a delivery task for a truck."""
    task_id: str
    origin: str  # Node ID
    destination: str  # Node ID
    demand: float = 1.0  # Load units
    priority: int = 1  # Higher = more urgent
    time_window: Optional[Tuple[float, float]] = None  # (earliest, latest)
    status: str = "pending"  # pending, in_progress, completed, failed
    

@dataclass
class TruckState:
    """Detailed state of a truck agent."""
    current_node: str
    current_edge: Optional[Tuple[str, str]] = None
    position_on_edge: float = 0.0  # 0.0 to 1.0
    current_route: List[str] = field(default_factory=list)
    current_task: Optional[DeliveryTask] = None
    cargo_load: float = 0.0
    cargo_capacity: float = 1.0
    speed: float = 0.0  # Current speed in km/h
    heading: Optional[str] = None  # Next node in route


class TruckAgent(BaseAgent):
    """
    Truck agent that navigates the road network and completes deliveries.
    
    Implements realistic truck behavior including:
    - Route planning and navigation
    - Speed adjustment based on traffic
    - Loading and unloading operations
    - Multiple routing policies
    
    Attributes:
        truck_type: Type of truck (small/medium/large)
        routing_policy: Strategy for route selection
        state: Detailed truck state including position and cargo
    """
    
    # Default truck type configurations
    TRUCK_CONFIGS = {
        TruckType.SMALL: {
            "max_speed": 60.0,  # km/h
            "capacity": 5.0,   # load units
            "fuel_rate": 0.08, # L/km
            "length": 6.0,     # meters
        },
        TruckType.MEDIUM: {
            "max_speed": 50.0,
            "capacity": 10.0,
            "fuel_rate": 0.12,
            "length": 10.0,
        },
        TruckType.LARGE: {
            "max_speed": 40.0,
            "capacity": 20.0,
            "fuel_rate": 0.18,
            "length": 16.0,
        }
    }
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        truck_type: TruckType = TruckType.MEDIUM,
        routing_policy: RoutingPolicy = RoutingPolicy.CONGESTION_AWARE,
        start_node: Optional[str] = None,
        depot_node: Optional[str] = None,
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a truck agent.
        
        Args:
            agent_id: Optional custom agent ID
            name: Human-readable name
            truck_type: Type of truck with specific characteristics
            routing_policy: Route selection strategy
            start_node: Starting node ID in the network
            depot_node: Home depot node ID for return trips
            verbose: Whether to print detailed logs
            config: Optional custom configuration overriding defaults
        """
        super().__init__(agent_id, name, verbose)
        
        self.truck_type = truck_type
        self.routing_policy = routing_policy
        self.metrics = TruckMetrics()
        
        # Load configuration
        self.config = self.TRUCK_CONFIGS[truck_type].copy()
        if config:
            self.config.update(config)
        
        # Initialize state
        self.state = AgentState.IDLE
        self.truck_state = TruckState(
            current_node=start_node or "",
            cargo_capacity=self.config["capacity"]
        )
        
        self.depot_node = depot_node or start_node
        self._speed_history: List[float] = []
        self._distance_history: List[float] = []
        
    def __repr__(self) -> str:
        return (f"TruckAgent(id={self.agent_id}, type={self.truck_type.value}, "
                f"state={self.state.value}, node={self.truck_state.current_node})")
    
    @property
    def position(self) -> Tuple[str, Optional[Tuple[str, str]], float]:
        """Get current position as (node, edge, position_on_edge)."""
        return (
            self.truck_state.current_node,
            self.truck_state.current_edge,
            self.truck_state.position_on_edge
        )
    
    @property
    def current_speed(self) -> float:
        """Get current speed in km/h."""
        return self.truck_state.speed
    
    @property
    def cargo_utilization(self) -> float:
        """Get cargo utilization as a fraction of capacity."""
        return self.truck_state.cargo_load / self.truck_state.cargo_capacity
    
    def assign_task(self, task: DeliveryTask) -> bool:
        """
        Assign a delivery task to the truck.
        
        Args:
            task: The delivery task to assign
            
        Returns:
            True if task was accepted, False otherwise
        """
        if self.truck_state.current_task is not None:
            return False
        
        if task.demand > self.truck_state.cargo_capacity - self.truck_state.cargo_load:
            return False
        
        self.truck_state.current_task = task
        task.status = "in_progress"
        self._log_action("task_assigned", {"task_id": task.task_id, "destination": task.destination})
        return True
    
    def calculate_route(
        self,
        destination: str,
        network: Any,
        traffic_model: Optional[Any] = None
    ) -> List[str]:
        """
        Calculate route to destination based on routing policy.
        
        Args:
            destination: Target node ID
            network: Road network object
            traffic_model: Optional traffic model for congestion-aware routing
            
        Returns:
            List of node IDs forming the route
        """
        origin = self.truck_state.current_node
        
        if self.routing_policy == RoutingPolicy.SHORTEST:
            route = network.shortest_path(origin, destination, weight="length")
            
        elif self.routing_policy == RoutingPolicy.FASTEST:
            route = network.shortest_path(origin, destination, weight="travel_time")
            
        elif self.routing_policy == RoutingPolicy.CONGESTION_AWARE:
            # Weight edges by both distance and congestion
            route = network.congestion_aware_path(
                origin, destination,
                congestion_data=traffic_model.get_congestion() if traffic_model else None
            )
            
        elif self.routing_policy == RoutingPolicy.ADAPTIVE:
            # Consider multiple factors dynamically
            route = network.adaptive_path(
                origin, destination,
                current_time=network.current_time,
                traffic_model=traffic_model
            )
        else:
            route = network.shortest_path(origin, destination)
        
        self.truck_state.current_route = route
        self._log_action("route_calculated", {
            "origin": origin,
            "destination": destination,
            "route_length": len(route),
            "policy": self.routing_policy.value
        })
        
        return route
    
    def step(self, environment: Any, dt: float = 1.0) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Args:
            environment: Simulation environment containing network and traffic
            dt: Time step in minutes
            
        Returns:
            Dictionary with step results
        """
        network = environment.network
        traffic = environment.traffic_model
        
        # Update based on current state
        if self.state == AgentState.IDLE:
            result = self._step_idle(environment, dt)
        elif self.state == AgentState.TRAVELING:
            result = self._step_traveling(environment, dt)
        elif self.state == AgentState.LOADING:
            result = self._step_loading(environment, dt)
        elif self.state == AgentState.UNLOADING:
            result = self._step_unloading(environment, dt)
        elif self.state == AgentState.WAITING:
            result = self._step_waiting(environment, dt)
        else:
            result = {"status": "no_action"}
        
        # Update metrics
        self.metrics.total_time += dt
        
        return result
    
    def _step_idle(self, environment: Any, dt: float) -> Dict[str, Any]:
        """Handle idle state - look for tasks or return to depot."""
        if self.truck_state.current_task:
            # Start heading to task origin
            task = self.truck_state.current_task
            
            if self.truck_state.current_node != task.origin:
                # Go to origin to load
                self.calculate_route(task.origin, environment.network)
                self.transition_to(AgentState.TRAVELING)
            else:
                # Already at origin, start loading
                self.transition_to(AgentState.LOADING)
            
            return {"status": "task_started"}
        
        return {"status": "idle"}
    
    def _step_traveling(self, environment: Any, dt: float) -> Dict[str, Any]:
        """Handle traveling state - move along route."""
        network = environment.network
        traffic = environment.traffic_model
        
        route = self.truck_state.current_route
        if not route or len(route) < 2:
            # Arrived at destination
            self._handle_arrival(environment)
            return {"status": "arrived"}
        
        # Get current edge
        current_node = route[0]
        next_node = route[1]
        edge = (current_node, next_node)
        
        # Calculate speed based on traffic
        edge_data = network.get_edge(edge)
        congestion = traffic.get_edge_congestion(edge) if traffic else 1.0
        
        max_speed = self.config["max_speed"]
        edge_speed_limit = edge_data.get("speed_limit", max_speed)
        
        # Speed reduction due to congestion
        actual_speed = min(max_speed, edge_speed_limit) * (1.0 - 0.7 * congestion)
        actual_speed = max(actual_speed, 5.0)  # Minimum speed
        
        self.truck_state.speed = actual_speed
        
        # Calculate distance traveled
        edge_length = edge_data.get("length", 1.0)  # km
        distance_traveled = actual_speed * (dt / 60.0)  # km
        
        # Update position on edge
        self.truck_state.position_on_edge += distance_traveled / edge_length
        
        # Check if reached end of edge
        if self.truck_state.position_on_edge >= 1.0:
            # Move to next node
            self.truck_state.current_node = next_node
            self.truck_state.position_on_edge = 0.0
            self.truck_state.current_route = route[1:]
            
            # Track congestion encountered
            if congestion > 0.3:
                self.metrics.congestion_encountered += dt
            
            self._log_action("node_reached", {
                "node": next_node,
                "speed": actual_speed,
                "congestion": congestion
            })
            
            # Check if this is the destination
            if len(self.truck_state.current_route) <= 1:
                self._handle_arrival(environment)
                return {"status": "arrived", "node": next_node}
        
        # Update metrics
        self.metrics.total_distance += distance_traveled
        self.metrics.fuel_consumed += distance_traveled * self.config["fuel_rate"]
        self._speed_history.append(actual_speed)
        
        return {
            "status": "traveling",
            "speed": actual_speed,
            "distance": distance_traveled,
            "congestion": congestion
        }
    
    def _step_loading(self, environment: Any, dt: float) -> Dict[str, Any]:
        """Handle loading cargo at origin."""
        task = self.truck_state.current_task
        if not task:
            self.transition_to(AgentState.IDLE)
            return {"status": "no_task"}
        
        # Simulate loading time
        loading_time_needed = task.demand * 5  # 5 minutes per unit
        
        if not hasattr(self, '_loading_progress'):
            self._loading_progress = 0.0
        
        self._loading_progress += dt
        
        if self._loading_progress >= loading_time_needed:
            # Loading complete
            self.truck_state.cargo_load = task.demand
            self._loading_progress = 0.0
            
            # Calculate route to destination
            self.calculate_route(task.destination, environment.network, environment.traffic_model)
            self.transition_to(AgentState.TRAVELING)
            
            self._log_action("loading_complete", {"cargo": task.demand})
            return {"status": "loaded", "cargo": task.demand}
        
        return {"status": "loading", "progress": self._loading_progress / loading_time_needed}
    
    def _step_unloading(self, environment: Any, dt: float) -> Dict[str, Any]:
        """Handle unloading cargo at destination."""
        task = self.truck_state.current_task
        if not task:
            self.transition_to(AgentState.IDLE)
            return {"status": "no_task"}
        
        # Simulate unloading time
        unloading_time_needed = self.truck_state.cargo_load * 5
        
        if not hasattr(self, '_unloading_progress'):
            self._unloading_progress = 0.0
        
        self._unloading_progress += dt
        
        if self._unloading_progress >= unloading_time_needed:
            # Unloading complete
            task.status = "completed"
            self.metrics.deliveries_completed += 1
            self.truck_state.cargo_load = 0.0
            self._unloading_progress = 0.0
            self.truck_state.current_task = None
            
            self.transition_to(AgentState.IDLE)
            
            self._log_action("delivery_completed", {
                "task_id": task.task_id,
                "destination": task.destination
            })
            return {"status": "delivered", "task_id": task.task_id}
        
        return {"status": "unloading", "progress": self._unloading_progress / unloading_time_needed}
    
    def _step_waiting(self, environment: Any, dt: float) -> Dict[str, Any]:
        """Handle waiting state (e.g., at traffic signals)."""
        self.metrics.waiting_time += dt
        return {"status": "waiting"}
    
    def _handle_arrival(self, environment: Any) -> None:
        """Handle arrival at a destination."""
        task = self.truck_state.current_task
        
        if task and self.truck_state.current_node == task.destination:
            # Arrived at delivery destination
            self.transition_to(AgentState.UNLOADING)
        elif task and self.truck_state.current_node == task.origin:
            # Arrived at pickup location
            self.transition_to(AgentState.LOADING)
        elif self.depot_node and self.truck_state.current_node == self.depot_node:
            # Returned to depot
            self.transition_to(AgentState.IDLE)
        else:
            # At intermediate point
            self.transition_to(AgentState.IDLE)
    
    def decide(self, context: Dict[str, Any]) -> Any:
        """
        Make a decision based on context.
        
        For trucks, this primarily involves routing decisions.
        
        Args:
            context: Dictionary containing network state, traffic info, etc.
            
        Returns:
            Decision (route choice, wait, etc.)
        """
        # Extract relevant context
        current_congestion = context.get("congestion", {})
        alternatives = context.get("route_alternatives", [])
        
        if self.routing_policy == RoutingPolicy.ADAPTIVE and alternatives:
            # Choose best alternative based on current conditions
            best_route = min(alternatives, key=lambda r: self._evaluate_route(r, current_congestion))
            return best_route
        
        return None
    
    def _evaluate_route(self, route: List[str], congestion: Dict) -> float:
        """Evaluate a route based on multiple factors."""
        # Simplified evaluation - in practice would use more sophisticated scoring
        total_cost = 0.0
        
        for i in range(len(route) - 1):
            edge = (route[i], route[i + 1])
            edge_congestion = congestion.get(edge, 0.0)
            total_cost += 1.0 + edge_congestion * 2.0  # Penalize congestion
        
        return total_cost
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the truck's current state."""
        return {
            "agent_id": self.agent_id,
            "truck_type": self.truck_type.value,
            "state": self.state.value,
            "current_node": self.truck_state.current_node,
            "speed": self.truck_state.speed,
            "cargo_load": self.truck_state.cargo_load,
            "cargo_capacity": self.truck_state.cargo_capacity,
            "cargo_utilization": self.cargo_utilization,
            "current_task": self.truck_state.current_task.task_id if self.truck_state.current_task else None,
            "route_length": len(self.truck_state.current_route),
            "metrics": self.get_metrics()
        }
    
    def finalize_metrics(self) -> None:
        """Calculate final aggregate metrics at end of simulation."""
        if self._speed_history:
            self.metrics.avg_speed = sum(self._speed_history) / len(self._speed_history)
        
        if self.truck_state.current_task:
            self.truck_state.current_task.status = "failed"
            self.metrics.deliveries_failed += 1
