"""
Agent-Based Truck Productivity Simulation
==========================================

A multi-agent simulation framework for analyzing truck fleet productivity
under traffic congestion constraints.

Modules:
    - agents: Agent definitions and behavior models
    - environment: Road network and traffic dynamics
    - optimization: Parameter sweep and optimization tools
    - utils: Visualization and metrics utilities
"""

__version__ = "1.0.0"
__author__ = "Afshin Mohammadi"

from src.simulation import Simulation
from src.agents import TruckAgent, TrafficAgent
from src.environment import RoadNetwork, TrafficModel

__all__ = [
    "Simulation",
    "TruckAgent",
    "TrafficAgent", 
    "RoadNetwork",
    "TrafficModel",
]
