"""
Traffic Agent Module
====================

Implements network-level traffic management agents that control
traffic signals, monitor congestion, and coordinate flow.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from src.agents.base_agent import BaseAgent, AgentState


class ControlStrategy(Enum):
    """Traffic control strategies."""
    FIXED = "fixed"              # Fixed timing
    ADAPTIVE = "adaptive"         # Adaptive to traffic
    COORDINATED = "coordinated"   # Coordinated across network


@dataclass
class SignalState:
    """State of a traffic signal."""
    node_id: str
    current_phase: int = 0
    time_in_phase: float = 0.0
    phase_durations: List[float] = field(default_factory=lambda: [30.0, 30.0])
    
    @property
    def n_phases(self) -> int:
        return len(self.phase_durations)


class TrafficAgent(BaseAgent):
    """
    Agent responsible for traffic management at intersections.
    
    Controls traffic signals and monitors congestion to optimize
    traffic flow through the network.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        control_strategy: ControlStrategy = ControlStrategy.ADAPTIVE,
        managed_nodes: Optional[List[str]] = None,
        verbose: bool = False
    ):
        super().__init__(agent_id, name, verbose)
        
        self.control_strategy = control_strategy
        self.managed_nodes = managed_nodes or []
        self.signals: Dict[str, SignalState] = {}
        
        # Initialize signals for managed nodes
        for node in self.managed_nodes:
            self.signals[node] = SignalState(node_id=node)
    
    def step(self, environment: Any, dt: float = 1.0) -> Dict[str, Any]:
        """Execute one simulation step."""
        results = {}
        
        for node_id, signal in self.signals.items():
            signal.time_in_phase += dt
            
            # Check if phase should change
            if signal.time_in_phase >= signal.phase_durations[signal.current_phase]:
                self._change_phase(signal, environment)
            
            results[node_id] = {
                "phase": signal.current_phase,
                "time_remaining": signal.phase_durations[signal.current_phase] - signal.time_in_phase
            }
        
        return results
    
    def _change_phase(self, signal: SignalState, environment: Any) -> None:
        """Change signal phase based on control strategy."""
        signal.time_in_phase = 0.0
        signal.current_phase = (signal.current_phase + 1) % signal.n_phases
        
        if self.control_strategy == ControlStrategy.ADAPTIVE:
            self._adapt_timing(signal, environment)
        
        self._log_action("phase_change", {
            "node": signal.node_id,
            "new_phase": signal.current_phase
        })
    
    def _adapt_timing(self, signal: SignalState, environment: Any) -> None:
        """Adapt signal timing based on current traffic."""
        traffic = environment.traffic_model
        
        # Get congestion for incoming edges
        node = signal.node_id
        incoming_edges = environment.network.get_incoming_edges(node)
        
        congestion = [
            traffic.get_edge_congestion(edge)
            for edge in incoming_edges
        ]
        
        if congestion:
            avg_congestion = sum(congestion) / len(congestion)
            
            # Adjust phase duration based on congestion
            base_duration = 30.0
            adjustment = avg_congestion * 30.0  # Up to 30s extra
            
            signal.phase_durations[signal.current_phase] = base_duration + adjustment
    
    def decide(self, context: Dict[str, Any]) -> Any:
        """Make traffic control decision."""
        if self.control_strategy == ControlStrategy.COORDINATED:
            return self._coordinated_decision(context)
        return None
    
    def _coordinated_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make coordinated decision across multiple intersections."""
        # Implement green wave coordination
        decisions = {}
        
        for node_id, signal in self.signals.items():
            decisions[node_id] = {
                "phase": signal.current_phase,
                "duration": signal.phase_durations[signal.current_phase]
            }
        
        return decisions
    
    def add_managed_node(self, node_id: str, phase_durations: Optional[List[float]] = None) -> None:
        """Add a node to be managed by this agent."""
        self.managed_nodes.append(node_id)
        self.signals[node_id] = SignalState(
            node_id=node_id,
            phase_durations=phase_durations or [30.0, 30.0]
        )
