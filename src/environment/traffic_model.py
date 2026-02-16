"""
Traffic Model Module
====================

Implements traffic flow dynamics using a modified Cell Transmission Model (CTM)
with realistic congestion propagation, shockwave effects, and queue dynamics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import random

import numpy as np


@dataclass
class TrafficState:
    """State of traffic on a single edge."""
    density: float = 0.0  # vehicles per km per lane
    flow: float = 0.0     # vehicles per hour
    speed: float = 40.0   # km/h
    queue_length: float = 0.0  # km
    
    @property
    def congestion_level(self) -> float:
        """Calculate congestion level (0 = free flow, 1 = jam)."""
        # Normalize density to 0-1 range
        # Using triangular fundamental diagram assumption
        critical_density = 25.0  # vehicles/km/lane
        jam_density = 150.0  # vehicles/km/lane
        
        if self.density <= critical_density:
            return self.density / critical_density * 0.5
        else:
            return 0.5 + 0.5 * (self.density - critical_density) / (jam_density - critical_density)


@dataclass
class FundamentalDiagramParams:
    """Parameters for the fundamental diagram (flow-density relationship)."""
    free_flow_speed: float = 40.0  # km/h
    jam_density: float = 150.0     # vehicles/km/lane
    capacity: float = 1800.0       # vehicles/hour/lane
    
    @property
    def critical_density(self) -> float:
        """Density at maximum flow."""
        return self.capacity / self.free_flow_speed


class TrafficModel:
    """
    Traffic flow model using modified Cell Transmission Model.
    
    Simulates:
    - Flow propagation based on density
    - Congestion formation and dissipation
    - Shockwave propagation
    - Queue spillback effects
    
    Attributes:
        network: Reference to the road network
        edge_states: Traffic state for each edge
        time_step: Simulation time step in minutes
    """
    
    def __init__(
        self,
        network: Any,
        time_step: float = 1.0,
        params: Optional[FundamentalDiagramParams] = None
    ):
        """
        Initialize the traffic model.
        
        Args:
            network: Road network object
            time_step: Simulation time step in minutes
            params: Fundamental diagram parameters
        """
        self.network = network
        self.time_step = time_step
        self.params = params or FundamentalDiagramParams()
        
        # Initialize edge states
        self.edge_states: Dict[Tuple[str, str], TrafficState] = {}
        self._initialize_states()
        
        # Demand patterns (can be time-dependent)
        self.base_demand: Dict[Tuple[str, str], float] = {}
        
        # Historical data for analysis
        self.history: List[Dict[str, TrafficState]] = []
        
    def _initialize_states(self) -> None:
        """Initialize traffic states for all edges."""
        for source, target, data in self.network.graph.edges(data=True):
            edge = (source, target)
            self.edge_states[edge] = TrafficState(
                speed=data.get('free_flow_speed', self.params.free_flow_speed),
                density=0.0,
                flow=0.0
            )
    
    def set_base_demand(self, demand_pattern: Dict[Tuple[str, str], float]) -> None:
        """
        Set base demand pattern (vehicles/hour).
        
        Args:
            demand_pattern: Dictionary mapping edges to demand values
        """
        self.base_demand = demand_pattern
    
    def get_edge_congestion(self, edge: Tuple[str, str]) -> float:
        """
        Get congestion level for an edge (0-1).
        
        Args:
            edge: Tuple of (source, target) node IDs
            
        Returns:
            Congestion level between 0 (free flow) and 1 (jam)
        """
        if edge in self.edge_states:
            return self.edge_states[edge].congestion_level
        return 0.0
    
    def get_congestion(self) -> Dict[Tuple[str, str], float]:
        """
        Get congestion levels for all edges.
        
        Returns:
            Dictionary mapping edges to congestion levels
        """
        return {
            edge: state.congestion_level
            for edge, state in self.edge_states.items()
        }
    
    def get_edge_speed(self, edge: Tuple[str, str]) -> float:
        """
        Get current speed on an edge.
        
        Args:
            edge: Tuple of (source, target) node IDs
            
        Returns:
            Current speed in km/h
        """
        if edge in self.edge_states:
            return self.edge_states[edge].speed
        return self.params.free_flow_speed
    
    def update(self, vehicles: List[Any], dt: float = 1.0) -> None:
        """
        Update traffic state based on vehicle positions.
        
        Args:
            vehicles: List of vehicles on the network
            dt: Time step in minutes
        """
        # Count vehicles on each edge
        edge_counts: Dict[Tuple[str, str], int] = {edge: 0 for edge in self.edge_states}
        
        for vehicle in vehicles:
            if hasattr(vehicle, 'truck_state') and vehicle.truck_state.current_edge:
                edge = vehicle.truck_state.current_edge
                if edge in edge_counts:
                    edge_counts[edge] += 1
        
        # Update traffic states
        for edge, count in edge_counts.items():
            self._update_edge_state(edge, count, dt)
        
        # Apply congestion propagation effects
        self._propagate_congestion(dt)
        
        # Store history
        self._record_state()
    
    def _update_edge_state(
        self,
        edge: Tuple[str, str],
        vehicle_count: int,
        dt: float
    ) -> None:
        """Update traffic state for a single edge."""
        edge_data = self.network.get_edge(edge)
        state = self.edge_states[edge]
        
        # Calculate density (vehicles per km per lane)
        length = edge_data.get('length', 1.0)
        lanes = edge_data.get('lanes', 1)
        
        if length > 0 and lanes > 0:
            state.density = vehicle_count / (length * lanes)
        
        # Calculate flow and speed using fundamental diagram
        state.flow, state.speed = self._fundamental_diagram(state.density, edge_data)
        
        # Update queue length if congested
        if state.speed < edge_data.get('free_flow_speed', self.params.free_flow_speed) * 0.5:
            state.queue_length = min(length, state.density * length * lanes / self.params.jam_density)
        else:
            state.queue_length = max(0, state.queue_length - dt / 60.0)
    
    def _fundamental_diagram(
        self,
        density: float,
        edge_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Calculate flow and speed from density using triangular fundamental diagram.
        
        Args:
            density: Traffic density (vehicles/km/lane)
            edge_data: Edge attributes
            
        Returns:
            Tuple of (flow, speed)
        """
        free_flow_speed = edge_data.get('free_flow_speed', self.params.free_flow_speed)
        jam_density = self.params.jam_density
        critical_density = self.params.critical_density
        
        lanes = edge_data.get('lanes', 1)
        
        if density <= critical_density:
            # Free flow branch
            speed = free_flow_speed
            flow = density * speed * lanes
        else:
            # Congested branch
            if density < jam_density:
                # Linear decrease in speed
                speed = free_flow_speed * (jam_density - density) / (jam_density - critical_density)
                speed = max(speed, free_flow_speed * 0.1)  # Minimum speed
            else:
                speed = free_flow_speed * 0.1  # Minimum speed in jam
            
            flow = density * speed * lanes
        
        return flow, speed
    
    def _propagate_congestion(self, dt: float) -> None:
        """Propagate congestion effects to upstream edges."""
        propagation_factor = 0.1 * (dt / 60.0)  # Scale by time step
        
        for edge, state in self.edge_states.items():
            if state.congestion_level > 0.5:
                # Congested edge - propagate upstream
                source = edge[0]
                
                # Get upstream edges
                for upstream_edge in self.network.get_incoming_edges(source):
                    if upstream_edge in self.edge_states:
                        upstream_state = self.edge_states[upstream_edge]
                        
                        # Increase density upstream
                        density_increase = (state.density - upstream_state.density) * propagation_factor
                        upstream_state.density = min(
                            upstream_state.density + density_increase,
                            self.params.jam_density
                        )
                        
                        # Update flow and speed
                        upstream_state.flow, upstream_state.speed = self._fundamental_diagram(
                            upstream_state.density,
                            self.network.get_edge(upstream_edge)
                        )
    
    def _record_state(self) -> None:
        """Record current state for history."""
        state_snapshot = {
            edge: TrafficState(
                density=s.density,
                flow=s.flow,
                speed=s.speed,
                queue_length=s.queue_length
            )
            for edge, s in self.edge_states.items()
        }
        self.history.append(state_snapshot)
    
    def predict_congestion(self, future_time: float) -> Dict[Tuple[str, str], float]:
        """
        Predict congestion levels at a future time.
        
        Simple prediction based on current trends.
        
        Args:
            future_time: Time to predict for (simulation minutes)
            
        Returns:
            Predicted congestion levels
        """
        if len(self.history) < 2:
            return self.get_congestion()
        
        # Simple linear extrapolation
        current = self.history[-1]
        previous = self.history[-2]
        
        predictions = {}
        time_diff = 1.0  # Assuming 1 minute between history records
        extrapolation_factor = (future_time - self.network.current_time) / time_diff
        
        for edge in current:
            current_congestion = current[edge].congestion_level
            previous_congestion = previous[edge].congestion_level
            
            trend = current_congestion - previous_congestion
            predicted = current_congestion + trend * extrapolation_factor
            
            predictions[edge] = max(0.0, min(1.0, predicted))
        
        return predictions
    
    def get_average_speed(self) -> float:
        """Get network-wide average speed."""
        speeds = [s.speed for s in self.edge_states.values()]
        return sum(speeds) / len(speeds) if speeds else self.params.free_flow_speed
    
    def get_total_delay(self) -> float:
        """Calculate total delay due to congestion (vehicle-hours)."""
        total_delay = 0.0
        
        for edge, state in self.edge_states.items():
            edge_data = self.network.get_edge(edge)
            free_flow_speed = edge_data.get('free_flow_speed', self.params.free_flow_speed)
            
            if state.speed < free_flow_speed:
                # Calculate delay
                delay_per_vehicle = (free_flow_speed - state.speed) / free_flow_speed
                vehicles = state.density * edge_data.get('length', 1.0) * edge_data.get('lanes', 1)
                total_delay += delay_per_vehicle * vehicles
        
        return total_delay
    
    def get_network_congestion_summary(self) -> Dict[str, Any]:
        """Get summary of network congestion."""
        congestion_levels = [s.congestion_level for s in self.edge_states.values()]
        
        return {
            "avg_congestion": sum(congestion_levels) / len(congestion_levels) if congestion_levels else 0,
            "max_congestion": max(congestion_levels) if congestion_levels else 0,
            "edges_congested": sum(1 for c in congestion_levels if c > 0.5),
            "edges_jammed": sum(1 for c in congestion_levels if c > 0.8),
            "avg_speed": self.get_average_speed(),
            "total_delay": self.get_total_delay(),
        }
    
    def reset(self) -> None:
        """Reset traffic states to initial conditions."""
        self._initialize_states()
        self.history.clear()
    
    def add_background_traffic(self, intensity: float = 0.3) -> None:
        """
        Add background traffic to simulate other vehicles.
        
        Args:
            intensity: Traffic intensity as fraction of capacity (0-1)
        """
        for edge, state in self.edge_states.items():
            edge_data = self.network.get_edge(edge)
            lanes = edge_data.get('lanes', 1)
            
            # Set background density
            background_density = intensity * self.params.critical_density * lanes
            state.density = background_density
            
            # Update speed based on density
            state.flow, state.speed = self._fundamental_diagram(
                state.density,
                edge_data
            )
