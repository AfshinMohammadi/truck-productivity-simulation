"""
Road Network Module
===================

Implements the road network as a directed graph with dynamic attributes
for traffic flow modeling. Supports various network topologies and
realistic road characteristics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import heapq
import math
import random

import networkx as nx
import numpy as np


class NetworkType(Enum):
    """Types of network topologies."""
    GRID = "grid"
    RADIAL = "radial"
    RANDOM = "random"
    CUSTOM = "custom"


@dataclass
class NodeAttributes:
    """Attributes for network nodes."""
    node_type: str = "intersection"  # intersection, depot, delivery_point
    x: float = 0.0
    y: float = 0.0
    capacity: float = float('inf')  # Vehicle handling capacity
    service_time: float = 0.0  # Average service time in minutes
    

@dataclass
class EdgeAttributes:
    """Attributes for network edges (road segments)."""
    length: float = 1.0  # km
    speed_limit: float = 40.0  # km/h
    capacity: float = 50.0  # vehicles per hour
    lanes: int = 1
    road_type: str = "local"  # local, arterial, highway
    free_flow_speed: float = 40.0  # km/h
    jam_density: float = 150.0  # vehicles per km per lane
    

class RoadNetwork:
    """
    Road network represented as a directed graph.
    
    Provides methods for:
    - Network creation and configuration
    - Path finding with various optimization criteria
    - Edge and node attribute management
    - Network analysis and statistics
    
    Attributes:
        graph: NetworkX directed graph representing the network
        depots: Set of depot node IDs
        delivery_points: Set of delivery point node IDs
    """
    
    def __init__(
        self,
        network_type: NetworkType = NetworkType.GRID,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the road network.
        
        Args:
            network_type: Type of network topology
            config: Configuration parameters for network generation
        """
        self.network_type = network_type
        self.config = config or {}
        self.graph = nx.DiGraph()
        self.depots: Set[str] = set()
        self.delivery_points: Set[str] = set()
        self.current_time: float = 0.0
        
        # Cache for computed paths
        self._path_cache: Dict[Tuple[str, str, str], List[str]] = {}
        
    @classmethod
    def from_config(cls, config_path: str) -> 'RoadNetwork':
        """
        Create network from YAML configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured RoadNetwork instance
        """
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        network = cls(
            network_type=NetworkType(config['network'].get('type', 'grid')),
            config=config
        )
        
        if config['network']['type'] == 'grid':
            network._create_grid_network(config['network'])
        elif config['network']['type'] == 'radial':
            network._create_radial_network(config['network'])
        
        # Add depots and delivery points
        network._add_special_nodes(config)
        
        return network
    
    def _create_grid_network(self, config: Dict[str, Any]) -> None:
        """Create a grid network."""
        dimensions = config.get('dimensions', [5, 5])
        edge_length = config.get('edge_length', 1.0)
        
        rows, cols = dimensions
        
        # Create nodes
        for i in range(rows):
            for j in range(cols):
                node_id = f"n_{i}_{j}"
                self.graph.add_node(
                    node_id,
                    **NodeAttributes(
                        node_type="intersection",
                        x=j * edge_length,
                        y=i * edge_length
                    ).__dict__
                )
        
        # Create edges (bidirectional)
        for i in range(rows):
            for j in range(cols):
                current = f"n_{i}_{j}"
                
                # Right neighbor
                if j < cols - 1:
                    neighbor = f"n_{i}_{j+1}"
                    self._add_edge_pair(current, neighbor, edge_length, config)
                
                # Down neighbor
                if i < rows - 1:
                    neighbor = f"n_{i+1}_{j}"
                    self._add_edge_pair(current, neighbor, edge_length, config)
    
    def _create_radial_network(self, config: Dict[str, Any]) -> None:
        """Create a radial/circular network."""
        center = config.get('center', (0, 0))
        radius = config.get('radius', 5.0)
        n_spokes = config.get('spokes', 8)
        n_rings = config.get('rings', 3)
        
        # Create center node
        self.graph.add_node(
            "center",
            **NodeAttributes(node_type="intersection", x=center[0], y=center[1]).__dict__
        )
        
        # Create rings and spokes
        for ring in range(1, n_rings + 1):
            ring_radius = radius * ring / n_rings
            
            for spoke in range(n_spokes):
                angle = 2 * math.pi * spoke / n_spokes
                x = center[0] + ring_radius * math.cos(angle)
                y = center[1] + ring_radius * math.sin(angle)
                
                node_id = f"r{ring}_s{spoke}"
                self.graph.add_node(
                    node_id,
                    **NodeAttributes(node_type="intersection", x=x, y=y).__dict__
                )
                
                # Connect to inner ring or center
                if ring == 1:
                    self._add_edge_pair("center", node_id, ring_radius, config)
                else:
                    inner_node = f"r{ring-1}_s{spoke}"
                    self._add_edge_pair(inner_node, node_id, radius / n_rings, config)
                
                # Connect within ring (circumferential)
                if spoke > 0:
                    prev_node = f"r{ring}_s{spoke-1}"
                    arc_length = ring_radius * 2 * math.pi / n_spokes
                    self._add_edge_pair(prev_node, node_id, arc_length, config)
                
                if spoke == n_spokes - 1:
                    first_node = f"r{ring}_s0"
                    arc_length = ring_radius * 2 * math.pi / n_spokes
                    self._add_edge_pair(node_id, first_node, arc_length, config)
    
    def _add_edge_pair(
        self,
        node1: str,
        node2: str,
        length: float,
        config: Dict[str, Any]
    ) -> None:
        """Add bidirectional edges between two nodes."""
        edge_config = config.get('edges', {})
        
        attrs1 = EdgeAttributes(
            length=length,
            speed_limit=edge_config.get('free_flow_speed', 40.0),
            capacity=edge_config.get('capacity', 50.0),
            lanes=edge_config.get('lanes', 1),
            road_type=edge_config.get('road_type', 'local'),
            free_flow_speed=edge_config.get('free_flow_speed', 40.0),
        )
        
        attrs2 = EdgeAttributes(
            length=length,
            speed_limit=edge_config.get('free_flow_speed', 40.0),
            capacity=edge_config.get('capacity', 50.0),
            lanes=edge_config.get('lanes', 1),
            road_type=edge_config.get('road_type', 'local'),
            free_flow_speed=edge_config.get('free_flow_speed', 40.0),
        )
        
        self.graph.add_edge(node1, node2, **attrs1.__dict__)
        self.graph.add_edge(node2, node1, **attrs2.__dict__)
    
    def _add_special_nodes(self, config: Dict[str, Any]) -> None:
        """Add depots and delivery points."""
        nodes_config = config.get('nodes', {})
        
        # Add depots
        depot_config = nodes_config.get('depot', {})
        n_depots = depot_config.get('count', 2)
        placement = depot_config.get('placement', 'corners')
        
        if placement == 'corners':
            all_nodes = list(self.graph.nodes())
            corners = [
                all_nodes[0],  # Top-left
                all_nodes[-1],  # Bottom-right
            ]
            for i in range(min(n_depots, len(corners))):
                self.depots.add(corners[i])
                self.graph.nodes[corners[i]]['node_type'] = 'depot'
        
        # Add delivery points
        delivery_config = nodes_config.get('delivery', {})
        n_delivery = delivery_config.get('count', 20)
        placement = delivery_config.get('placement', 'random')
        
        all_nodes = list(set(self.graph.nodes()) - self.depots)
        
        if placement == 'random':
            selected = random.sample(all_nodes, min(n_delivery, len(all_nodes)))
            for node in selected:
                self.delivery_points.add(node)
                self.graph.nodes[node]['node_type'] = 'delivery_point'
    
    def get_edge(self, edge: Tuple[str, str]) -> Dict[str, Any]:
        """
        Get attributes of an edge.
        
        Args:
            edge: Tuple of (source, target) node IDs
            
        Returns:
            Dictionary of edge attributes
        """
        if self.graph.has_edge(*edge):
            return dict(self.graph.edges[*edge])
        return {}
    
    def get_node(self, node: str) -> Dict[str, Any]:
        """
        Get attributes of a node.
        
        Args:
            node: Node ID
            
        Returns:
            Dictionary of node attributes
        """
        if self.graph.has_node(node):
            return dict(self.graph.nodes[node])
        return {}
    
    def shortest_path(
        self,
        source: str,
        target: str,
        weight: str = "length"
    ) -> List[str]:
        """
        Find shortest path between two nodes.
        
        Args:
            source: Starting node ID
            target: Destination node ID
            weight: Edge attribute to minimize ('length', 'travel_time', etc.)
            
        Returns:
            List of node IDs forming the path
        """
        cache_key = (source, target, weight)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key].copy()
        
        try:
            path = nx.shortest_path(self.graph, source, target, weight=weight)
            self._path_cache[cache_key] = path
            return path
        except nx.NetworkXNoPath:
            return []
    
    def congestion_aware_path(
        self,
        source: str,
        target: str,
        congestion_data: Optional[Dict[Tuple[str, str], float]] = None
    ) -> List[str]:
        """
        Find path considering current congestion levels.
        
        Uses a weighted combination of distance and congestion.
        
        Args:
            source: Starting node ID
            target: Destination node ID
            congestion_data: Dictionary mapping edges to congestion levels (0-1)
            
        Returns:
            List of node IDs forming the path
        """
        if congestion_data is None:
            return self.shortest_path(source, target)
        
        # Create temporary weight function
        def weight_func(u, v, d):
            base_weight = d['length']
            congestion = congestion_data.get((u, v), 0.0)
            # Increase weight for congested edges
            return base_weight * (1.0 + congestion * 2.0)
        
        try:
            path = nx.dijkstra_path(self.graph, source, target, weight=weight_func)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def adaptive_path(
        self,
        source: str,
        target: str,
        current_time: float,
        traffic_model: Optional[Any] = None
    ) -> List[str]:
        """
        Find path using adaptive routing considering predicted traffic.
        
        Args:
            source: Starting node ID
            target: Destination node ID
            current_time: Current simulation time
            traffic_model: Traffic model for predictions
            
        Returns:
            List of node IDs forming the path
        """
        # Get predicted congestion from traffic model
        if traffic_model and hasattr(traffic_model, 'predict_congestion'):
            predicted_congestion = traffic_model.predict_congestion(current_time + 30)  # 30 min ahead
            return self.congestion_aware_path(source, target, predicted_congestion)
        
        return self.shortest_path(source, target)
    
    def get_alternative_paths(
        self,
        source: str,
        target: str,
        k: int = 3
    ) -> List[List[str]]:
        """
        Get k alternative paths between two nodes.
        
        Args:
            source: Starting node ID
            target: Destination node ID
            k: Number of alternative paths to find
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        try:
            paths = list(nx.shortest_simple_paths(self.graph, source, target, weight="length"))
            return paths[:k]
        except nx.NetworkXNoPath:
            return []
    
    def get_neighbors(self, node: str) -> List[str]:
        """Get neighboring nodes (outgoing edges)."""
        return list(self.graph.successors(node))
    
    def get_predecessors(self, node: str) -> List[str]:
        """Get predecessor nodes (incoming edges)."""
        return list(self.graph.predecessors(node))
    
    def get_outgoing_edges(self, node: str) -> List[Tuple[str, str]]:
        """Get all outgoing edges from a node."""
        return [(node, n) for n in self.graph.successors(node)]
    
    def get_incoming_edges(self, node: str) -> List[Tuple[str, str]]:
        """Get all incoming edges to a node."""
        return [(n, node) for n in self.graph.predecessors(node)]
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "n_depots": len(self.depots),
            "n_delivery_points": len(self.delivery_points),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            "total_road_length": sum(d['length'] for _, _, d in self.graph.edges(data=True)),
            "is_connected": nx.is_strongly_connected(self.graph),
        }
    
    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get positions of all nodes for visualization."""
        return {
            node: (data.get('x', 0), data.get('y', 0))
            for node, data in self.graph.nodes(data=True)
        }
    
    def clear_cache(self) -> None:
        """Clear the path cache."""
        self._path_cache.clear()
    
    def update_time(self, dt: float) -> None:
        """Update current simulation time."""
        self.current_time += dt
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize network to dictionary."""
        return {
            "network_type": self.network_type.value,
            "nodes": {n: dict(d) for n, d in self.graph.nodes(data=True)},
            "edges": [(*e, dict(d)) for e, d in self.graph.edges(data=True)],
            "depots": list(self.depots),
            "delivery_points": list(self.delivery_points),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoadNetwork':
        """Create network from dictionary."""
        network = cls(network_type=NetworkType(data.get('network_type', 'grid')))
        
        # Add nodes
        for node_id, attrs in data['nodes'].items():
            network.graph.add_node(node_id, **attrs)
        
        # Add edges
        for source, target, attrs in data['edges']:
            network.graph.add_edge(source, target, **attrs)
        
        network.depots = set(data.get('depots', []))
        network.delivery_points = set(data.get('delivery_points', []))
        
        return network
