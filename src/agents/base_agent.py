"""
Base Agent Module
=================

Defines the abstract base class for all agents in the simulation.
Implements common functionality for state management, logging, and
inter-agent communication.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid
import time


class AgentState(Enum):
    """Enumeration of possible agent states."""
    IDLE = "idle"
    ACTIVE = "active"
    TRAVELING = "traveling"
    LOADING = "loading"
    UNLOADING = "unloading"
    WAITING = "waiting"
    COMPLETED = "completed"


@dataclass
class AgentMetrics:
    """Container for agent performance metrics."""
    total_distance: float = 0.0
    total_time: float = 0.0
    tasks_completed: int = 0
    waiting_time: float = 0.0
    fuel_consumed: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "total_distance": self.total_distance,
            "total_time": self.total_time,
            "tasks_completed": self.tasks_completed,
            "waiting_time": self.waiting_time,
            "fuel_consumed": self.fuel_consumed,
        }


@dataclass
class AgentLog:
    """Log entry for agent actions."""
    timestamp: float
    action: str
    details: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the simulation.
    
    Provides common functionality for:
    - Unique agent identification
    - State management and transitions
    - Metrics collection
    - Action logging
    - Inter-agent communication
    
    Attributes:
        agent_id: Unique identifier for the agent
        state: Current state of the agent
        metrics: Performance metrics collected during simulation
        log: List of action logs
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Optional custom agent ID, auto-generated if None
            name: Human-readable name for the agent
            verbose: Whether to print detailed logs during simulation
        """
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.name = name or f"Agent_{self.agent_id}"
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()
        self.log: List[AgentLog] = []
        self.verbose = verbose
        self._creation_time = time.time()
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, state={self.state.value})"
    
    def __hash__(self) -> int:
        return hash(self.agent_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseAgent):
            return False
        return self.agent_id == other.agent_id
    
    def transition_to(self, new_state: AgentState) -> None:
        """
        Transition agent to a new state.
        
        Args:
            new_state: The target state to transition to
        """
        old_state = self.state
        self.state = new_state
        self._log_action(
            action="state_transition",
            details={"from": old_state.value, "to": new_state.value}
        )
        
        if self.verbose:
            print(f"[{self.name}] State: {old_state.value} -> {new_state.value}")
    
    def _log_action(self, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an agent action.
        
        Args:
            action: Description of the action
            details: Additional details about the action
        """
        entry = AgentLog(
            timestamp=time.time() - self._creation_time,
            action=action,
            details=details or {}
        )
        self.log.append(entry)
    
    def get_log(self) -> List[Dict[str, Any]]:
        """
        Get the agent's action log as a list of dictionaries.
        
        Returns:
            List of log entries as dictionaries
        """
        return [
            {
                "timestamp": entry.timestamp,
                "action": entry.action,
                "details": entry.details
            }
            for entry in self.log
        ]
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get the agent's performance metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        return self.metrics.to_dict()
    
    def reset(self) -> None:
        """Reset agent to initial state, clearing metrics and logs."""
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()
        self.log = []
        self._creation_time = time.time()
    
    @abstractmethod
    def step(self, environment: Any, dt: float = 1.0) -> None:
        """
        Execute one simulation step for the agent.
        
        This method must be implemented by all concrete agent classes.
        
        Args:
            environment: The simulation environment
            dt: Time step duration in simulation time units
        """
        pass
    
    @abstractmethod
    def decide(self, context: Dict[str, Any]) -> Any:
        """
        Make a decision based on the current context.
        
        This method implements the agent's decision-making logic.
        
        Args:
            context: Current state of the environment and relevant information
            
        Returns:
            The decision (e.g., action to take, route to follow)
        """
        pass
    
    def communicate(self, message: Dict[str, Any], recipients: List['BaseAgent']) -> None:
        """
        Send a message to other agents.
        
        Args:
            message: The message content
            recipients: List of agents to receive the message
        """
        for recipient in recipients:
            if hasattr(recipient, 'receive_message'):
                recipient.receive_message(message, sender=self)
        
        self._log_action(
            action="send_message",
            details={"recipients": [r.agent_id for r in recipients], "message": message}
        )
    
    def receive_message(self, message: Dict[str, Any], sender: 'BaseAgent') -> None:
        """
        Receive a message from another agent.
        
        Args:
            message: The message content
            sender: The agent that sent the message
        """
        self._log_action(
            action="receive_message",
            details={"sender": sender.agent_id, "message": message}
        )
        
        if self.verbose:
            print(f"[{self.name}] Received message from {sender.name}: {message}")
