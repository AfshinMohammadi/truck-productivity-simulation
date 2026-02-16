"""
Visualization Module
====================

Provides comprehensive visualization tools for simulation results,
network analysis, and parameter sweep exploration.
"""

from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


class Visualizer:
    """
    Visualization toolkit for simulation analysis.
    
    Provides methods for:
    - Network visualization with congestion overlay
    - Agent trajectory plotting
    - Performance metric charts
    - Parameter sweep result visualization
    """
    
    def __init__(self, results: Any, output_dir: str = "results/figures"):
        """
        Initialize visualizer.
        
        Args:
            results: Simulation or sweep results object
            output_dir: Directory to save figures
        """
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_network(
        self,
        network: Any,
        congestion: Optional[Dict] = None,
        truck_positions: Optional[List[Tuple]] = None,
        title: str = "Network Visualization",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the road network with optional overlays.
        
        Args:
            network: Road network object
            congestion: Dictionary of edge congestion levels
            truck_positions: List of truck positions
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get node positions
        pos = network.get_node_positions()
        
        # Draw edges with congestion coloring
        if congestion:
            # Create line segments with colors
            segments = []
            colors = []
            
            for edge in network.graph.edges():
                source, target = edge
                if source in pos and target in pos:
                    x1, y1 = pos[source]
                    x2, y2 = pos[target]
                    segments.append([(x1, y1), (x2, y2)])
                    
                    congestion_level = congestion.get(edge, 0)
                    colors.append(congestion_level)
            
            # Create colormap (green to red)
            cmap = LinearSegmentedColormap.from_list('congestion', ['green', 'yellow', 'red'])
            
            lc = LineCollection(segments, cmap=cmap, linewidths=2)
            lc.set_array(np.array(colors))
            ax.add_collection(lc)
            
            # Add colorbar
            cbar = plt.colorbar(lc, ax=ax, label='Congestion Level')
        else:
            # Draw edges in gray
            for edge in network.graph.edges():
                source, target = edge
                if source in pos and target in pos:
                    x1, y1 = pos[source]
                    x2, y2 = pos[target]
                    ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.5)
        
        # Draw nodes
        for node, (x, y) in pos.items():
            node_type = network.get_node(node).get('node_type', 'intersection')
            
            if node_type == 'depot':
                ax.scatter(x, y, c='blue', s=100, marker='s', zorder=5, label='Depot')
            elif node_type == 'delivery_point':
                ax.scatter(x, y, c='green', s=50, marker='o', zorder=4)
            else:
                ax.scatter(x, y, c='gray', s=30, marker='o', zorder=3, alpha=0.5)
        
        # Draw trucks
        if truck_positions:
            for pos_info in truck_positions:
                node, edge, pos_on_edge = pos_info
                if edge and edge[0] in pos and edge[1] in pos:
                    x1, y1 = pos[edge[0]]
                    x2, y2 = pos[edge[1]]
                    x = x1 + pos_on_edge * (x2 - x1)
                    y = y1 + pos_on_edge * (y2 - y1)
                    ax.scatter(x, y, c='red', s=40, marker='^', zorder=6)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        handles = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Depot'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Delivery Point'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='Truck'),
        ]
        ax.legend(handles=handles, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_productivity_timeline(
        self,
        metrics: Any,
        title: str = "Productivity Over Time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot productivity metrics over simulation time.
        
        Args:
            metrics: Simulation metrics object
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        time_steps = range(len(metrics.deliveries_over_time))
        
        # Deliveries over time
        ax = axes[0, 0]
        ax.plot(time_steps, metrics.deliveries_over_time, 'b-', linewidth=2)
        ax.fill_between(time_steps, 0, metrics.deliveries_over_time, alpha=0.3)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Deliveries')
        ax.set_title('Total Deliveries')
        ax.grid(True, alpha=0.3)
        
        # Average speed over time
        ax = axes[0, 1]
        ax.plot(time_steps, metrics.avg_speed_over_time, 'g-', linewidth=2)
        ax.axhline(y=np.mean(metrics.avg_speed_over_time), color='r', linestyle='--', label='Average')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Average Speed (km/h)')
        ax.set_title('Fleet Average Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Congestion over time
        ax = axes[1, 0]
        ax.plot(time_steps, metrics.congestion_over_time, 'r-', linewidth=2)
        ax.fill_between(time_steps, 0, metrics.congestion_over_time, alpha=0.3, color='red')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Average Congestion')
        ax.set_title('Network Congestion Level')
        ax.grid(True, alpha=0.3)
        
        # Delivery rate
        ax = axes[1, 1]
        if len(metrics.deliveries_over_time) > 1:
            delivery_rate = np.diff(metrics.deliveries_over_time)
            ax.plot(time_steps[1:], delivery_rate, 'purple', linewidth=2)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Deliveries per Step')
            ax.set_title('Delivery Rate')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_congestion_heatmap(
        self,
        network: Any,
        traffic_model: Any,
        title: str = "Network Congestion Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot congestion heatmap across the network.
        
        Args:
            network: Road network object
            traffic_model: Traffic model with current state
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        pos = network.get_node_positions()
        congestion = traffic_model.get_congestion()
        
        # Create heatmap data
        x_coords = [p[0] for p in pos.values()]
        y_coords = [p[1] for p in pos.values()]
        
        # Draw edges colored by congestion
        cmap = LinearSegmentedColormap.from_list('congestion', ['green', 'yellow', 'orange', 'red'])
        
        for edge, state in traffic_model.edge_states.items():
            source, target = edge
            if source in pos and target in pos:
                x1, y1 = pos[source]
                x2, y2 = pos[target]
                
                color = cmap(state.congestion_level)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)
        
        # Draw nodes
        ax.scatter(x_coords, y_coords, c='navy', s=20, zorder=5, alpha=0.7)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Congestion Level')
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_performance_boxplots(
        self,
        sweep_result: Any,
        metrics: List[str],
        group_by: Optional[str] = None,
        title: str = "Performance Distributions",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot boxplots of performance metrics.
        
        Args:
            sweep_result: Parameter sweep results
            metrics: List of metric names to plot
            group_by: Parameter to group results by
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        df = sweep_result.get_dataframe()
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            metric_col = metric if metric in df.columns else f"{metric}_mean"
            
            if group_by and group_by in df.columns:
                df.boxplot(column=metric_col, by=group_by, ax=ax)
                ax.set_title(f"{metric} by {group_by}")
                plt.suptitle('')
            else:
                df.boxplot(column=metric_col, ax=ax)
                ax.set_title(metric)
            
            ax.set_ylabel(metric)
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_tornado(
        self,
        sensitivities: Dict[str, float],
        title: str = "Parameter Sensitivity Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot tornado chart for sensitivity analysis.
        
        Args:
            sensitivities: Dictionary of parameter sensitivities
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by sensitivity
        sorted_sens = dict(sorted(sensitivities.items(), key=lambda x: x[1]))
        
        params = list(sorted_sens.keys())
        values = list(sorted_sens.values())
        
        # Create horizontal bar chart
        y_pos = range(len(params))
        ax.barh(y_pos, values, align='center', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel('Relative Sensitivity')
        ax.set_title(title)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_pareto_front(
        self,
        pareto_solutions: List[Dict],
        obj1: str,
        obj2: str,
        title: str = "Pareto Front",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Pareto front for multi-objective optimization.
        
        Args:
            pareto_solutions: List of Pareto-optimal solutions
            obj1: First objective name
            obj2: Second objective name
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = [s['objectives'][obj1] for s in pareto_solutions]
        y = [s['objectives'][obj2] for s in pareto_solutions]
        
        ax.scatter(x, y, c='red', s=100, zorder=5, label='Pareto Optimal')
        
        # Connect points
        sorted_points = sorted(zip(x, y))
        ax.plot([p[0] for p in sorted_points], [p[1] for p in sorted_points], 
                'r--', alpha=0.5)
        
        ax.set_xlabel(obj1)
        ax.set_ylabel(obj2)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_summary_report(
        self,
        output_path: str = "results/summary_report.txt"
    ) -> str:
        """
        Create a text summary report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Report content as string
        """
        report = []
        report.append("=" * 60)
        report.append("SIMULATION SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Add metrics summary
        if hasattr(self.results, 'metrics'):
            m = self.results.metrics
            report.append("KEY PERFORMANCE METRICS")
            report.append("-" * 40)
            report.append(f"Total Deliveries: {m.total_deliveries}")
            report.append(f"Failed Deliveries: {m.failed_deliveries}")
            report.append(f"Total Distance: {m.total_distance:.2f} km")
            report.append(f"Total Fuel Consumed: {m.total_fuel:.2f} L")
            report.append(f"Average Speed: {m.avg_speed:.2f} km/h")
            report.append(f"Average Trip Time: {m.avg_trip_time:.2f} min")
            report.append(f"Fleet Efficiency: {m.fleet_efficiency:.4f} deliveries/km")
            report.append("")
        
        report_content = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        return report_content


def plot_simulation_comparison(
    simulations: List[Any],
    labels: List[str],
    metrics: List[str],
    title: str = "Simulation Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare multiple simulation runs.
    
    Args:
        simulations: List of simulation results
        labels: Labels for each simulation
        metrics: Metrics to compare
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = []
        for sim in simulations:
            if hasattr(sim, 'metrics'):
                val = getattr(sim.metrics, metric, 0)
            else:
                val = sim.get(metric, 0)
            values.append(val)
        
        ax.bar(labels, values, color=['steelblue', 'coral', 'green', 'purple'][:len(labels)])
        ax.set_ylabel(metric)
        ax.set_title(metric)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
