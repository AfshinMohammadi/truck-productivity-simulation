"""
Parameter Sweep Module
======================

Implements high-dimensional parameter space exploration for systematic
analysis of simulation behavior across varying configurations.

Features:
- Full factorial design
- Latin Hypercube Sampling
- Sobol sequence sampling
- Parallel execution support
- Result aggregation and analysis
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import product
import random
import time
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    values: List[Any]
    parameter_type: str = "categorical"  # categorical, continuous, integer
    
    def sample_value(self) -> Any:
        """Get a random value from the parameter."""
        return random.choice(self.values)


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    parameters: Dict[str, List[Any]]
    sampling_method: str = "full_factorial"  # full_factorial, latin_hypercube, sobol, random
    n_samples: int = 100  # For random/sampling methods
    n_replicates: int = 1
    seed: Optional[int] = None
    parallel: bool = True
    n_workers: int = -1  # -1 for auto
    
    def __post_init__(self):
        if self.n_workers == -1:
            self.n_workers = mp.cpu_count()


@dataclass
class SimulationResult:
    """Result from a single simulation run."""
    config_id: int
    parameters: Dict[str, Any]
    replicate: int
    metrics: Dict[str, float]
    raw_output: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "parameters": self.parameters,
            "replicate": self.replicate,
            "metrics": self.metrics,
        }


@dataclass 
class SweepResult:
    """Aggregated results from parameter sweep."""
    results: List[SimulationResult] = field(default_factory=list)
    parameter_names: List[str] = field(default_factory=list)
    
    def add_result(self, result: SimulationResult) -> None:
        self.results.append(result)
    
    def get_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd
        
        rows = []
        for r in self.results:
            row = {**r.parameters, **r.metrics, "replicate": r.replicate, "config_id": r.config_id}
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def aggregate_by_config(self) -> Dict[int, Dict[str, Any]]:
        """Aggregate metrics across replicates."""
        from collections import defaultdict
        
        aggregated = defaultdict(lambda: {"metrics": defaultdict(list), "parameters": {}, "n_replicates": 0})
        
        for result in self.results:
            agg = aggregated[result.config_id]
            agg["parameters"] = result.parameters
            agg["n_replicates"] += 1
            
            for metric, value in result.metrics.items():
                agg["metrics"][metric].append(value)
        
        # Calculate statistics
        summary = {}
        for config_id, data in aggregated.items():
            stats = {"parameters": data["parameters"], "n_replicates": data["n_replicates"]}
            
            for metric, values in data["metrics"].items():
                values = [v for v in values if v is not None]
                if values:
                    stats[f"{metric}_mean"] = np.mean(values)
                    stats[f"{metric}_std"] = np.std(values)
                    stats[f"{metric}_min"] = np.min(values)
                    stats[f"{metric}_max"] = np.max(values)
            
            summary[config_id] = stats
        
        return summary
    
    def get_pareto_front(
        self,
        objectives: List[str],
        directions: Optional[List[str]] = None
    ) -> List[SimulationResult]:
        """
        Get Pareto-optimal configurations.
        
        Args:
            objectives: List of metric names to optimize
            directions: 'minimize' or 'maximize' for each objective
            
        Returns:
            List of non-dominated results
        """
        if directions is None:
            directions = ["maximize"] * len(objectives)
        
        aggregated = self.aggregate_by_config()
        
        # Convert to objective vectors
        configs = []
        for config_id, stats in aggregated.items():
            vec = []
            for obj, direction in zip(objectives, directions):
                value = stats.get(f"{obj}_mean", 0)
                if direction == "minimize":
                    value = -value
                vec.append(value)
            configs.append((config_id, np.array(vec), stats))
        
        # Find non-dominated solutions
        pareto_front = []
        for i, (id_i, vec_i, stats_i) in enumerate(configs):
            dominated = False
            for j, (id_j, vec_j, stats_j) in enumerate(configs):
                if i != j:
                    if np.all(vec_j >= vec_i) and np.any(vec_j > vec_i):
                        dominated = True
                        break
            
            if not dominated:
                pareto_front.append({
                    "config_id": id_i,
                    "parameters": stats_i["parameters"],
                    "objectives": {obj: stats_i.get(f"{obj}_mean") for obj in objectives}
                })
        
        return pareto_front
    
    def save(self, path: str) -> None:
        """Save results to file."""
        data = {
            "results": [r.to_dict() for r in self.results],
            "parameter_names": self.parameter_names
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> 'SweepResult':
        """Load results from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        result = cls(parameter_names=data["parameter_names"])
        for r in data["results"]:
            result.results.append(SimulationResult(
                config_id=r["config_id"],
                parameters=r["parameters"],
                replicate=r["replicate"],
                metrics=r["metrics"]
            ))
        
        return result


class ParameterSweep:
    """
    High-dimensional parameter sweep framework.
    
    Enables systematic exploration of simulation behavior across
    varying configurations with support for multiple sampling methods
    and parallel execution.
    
    Example:
        >>> params = {
        ...     'n_trucks': [20, 30, 40, 50],
        ...     'routing_policy': ['shortest', 'congestion_aware'],
        ...     'fleet_mix': [(0.7, 0.3), (0.5, 0.5)]
        ... }
        >>> sweep = ParameterSweep(params, n_replicates=10)
        >>> results = sweep.run(simulation_function)
    """
    
    def __init__(
        self,
        parameter_ranges: Dict[str, List[Any]],
        sampling_method: str = "full_factorial",
        n_samples: int = 100,
        n_replicates: int = 1,
        seed: Optional[int] = None,
        parallel: bool = True,
        n_workers: int = -1
    ):
        """
        Initialize parameter sweep.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to possible values
            sampling_method: Method for sampling the parameter space
            n_samples: Number of samples for random/sampling methods
            n_replicates: Number of replicate runs per configuration
            seed: Random seed for reproducibility
            parallel: Whether to run in parallel
            n_workers: Number of parallel workers (-1 for auto)
        """
        self.parameter_ranges = parameter_ranges
        self.sampling_method = sampling_method
        self.n_samples = n_samples
        self.n_replicates = n_replicates
        self.seed = seed
        self.parallel = parallel
        self.n_workers = n_workers if n_workers > 0 else mp.cpu_count()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate parameter specifications."""
        for name, values in self.parameter_ranges.items():
            if not values:
                raise ValueError(f"Parameter '{name}' has no values")
            if len(values) < 2 and self.sampling_method == "full_factorial":
                # Warning but allow
                pass
    
    def generate_configs(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter configurations to test.
        
        Returns:
            List of configuration dictionaries
        """
        if self.sampling_method == "full_factorial":
            return self._full_factorial()
        elif self.sampling_method == "latin_hypercube":
            return self._latin_hypercube()
        elif self.sampling_method == "sobol":
            return self._sobol_sequence()
        elif self.sampling_method == "random":
            return self._random_sampling()
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
    
    def _full_factorial(self) -> List[Dict[str, Any]]:
        """Generate full factorial design."""
        names = list(self.parameter_ranges.keys())
        value_lists = [self.parameter_ranges[n] for n in names]
        
        configs = []
        for combination in product(*value_lists):
            config = dict(zip(names, combination))
            configs.append(config)
        
        return configs
    
    def _latin_hypercube(self) -> List[Dict[str, Any]]:
        """Generate Latin Hypercube sampling."""
        n_dims = len(self.parameter_ranges)
        names = list(self.parameter_ranges.keys())
        
        sampler = qmc.LatinHypercube(d=n_dims, seed=self.seed)
        samples = sampler.random(n=self.n_samples)
        
        configs = []
        for sample in samples:
            config = {}
            for i, name in enumerate(names):
                values = self.parameter_ranges[name]
                idx = int(sample[i] * len(values))
                idx = min(idx, len(values) - 1)
                config[name] = values[idx]
            configs.append(config)
        
        return configs
    
    def _sobol_sequence(self) -> List[Dict[str, Any]]:
        """Generate Sobol sequence sampling."""
        n_dims = len(self.parameter_ranges)
        names = list(self.parameter_ranges.keys())
        
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=self.seed)
        samples = sampler.random(n=self.n_samples)
        
        configs = []
        for sample in samples:
            config = {}
            for i, name in enumerate(names):
                values = self.parameter_ranges[name]
                idx = int(sample[i] * len(values))
                idx = min(idx, len(values) - 1)
                config[name] = values[idx]
            configs.append(config)
        
        return configs
    
    def _random_sampling(self) -> List[Dict[str, Any]]:
        """Generate random sampling."""
        configs = []
        names = list(self.parameter_ranges.keys())
        
        for _ in range(self.n_samples):
            config = {}
            for name in names:
                values = self.parameter_ranges[name]
                config[name] = random.choice(values)
            configs.append(config)
        
        return configs
    
    def _run_single(
        self,
        config: Dict[str, Any],
        config_id: int,
        replicate: int,
        simulation_func: Callable
    ) -> SimulationResult:
        """Run a single simulation."""
        result = simulation_func(config)
        
        return SimulationResult(
            config_id=config_id,
            parameters=config,
            replicate=replicate,
            metrics=result.get("metrics", result) if isinstance(result, dict) else result
        )
    
    def run(
        self,
        simulation_func: Callable[[Dict[str, Any]], Dict[str, float]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SweepResult:
        """
        Run the parameter sweep.
        
        Args:
            simulation_func: Function that takes a config dict and returns metrics dict
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            SweepResult containing all simulation results
        """
        configs = self.generate_configs()
        total_runs = len(configs) * self.n_replicates
        
        result = SweepResult(parameter_names=list(self.parameter_ranges.keys()))
        
        completed = 0
        
        # Generate all tasks
        tasks = []
        for config_id, config in enumerate(configs):
            for replicate in range(self.n_replicates):
                tasks.append((config, config_id, replicate))
        
        start_time = time.time()
        
        if self.parallel and len(tasks) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for config, config_id, replicate in tasks:
                    future = executor.submit(
                        self._run_single,
                        config, config_id, replicate, simulation_func
                    )
                    futures.append(future)
                
                for future in futures:
                    sim_result = future.result()
                    result.add_result(sim_result)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total_runs)
        else:
            # Sequential execution
            for config, config_id, replicate in tasks:
                sim_result = self._run_single(config, config_id, replicate, simulation_func)
                result.add_result(sim_result)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total_runs)
        
        elapsed = time.time() - start_time
        print(f"Parameter sweep completed: {total_runs} runs in {elapsed:.2f}s ({total_runs/elapsed:.2f} runs/s)")
        
        return result
    
    def run_parallel(
        self,
        simulation_func: Callable,
        n_workers: Optional[int] = None
    ) -> SweepResult:
        """
        Run parameter sweep with parallel execution.
        
        Convenience method that forces parallel execution.
        
        Args:
            simulation_func: Function that takes config and returns metrics
            n_workers: Number of workers (overrides instance setting)
            
        Returns:
            SweepResult with all results
        """
        if n_workers:
            self.n_workers = n_workers
        self.parallel = True
        return self.run(simulation_func)
    
    def estimate_runtime(
        self,
        sample_runs: int = 5,
        simulation_func: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Estimate total sweep runtime.
        
        Args:
            sample_runs: Number of sample runs to average
            simulation_func: Function to test (uses dummy if None)
            
        Returns:
            Dictionary with timing estimates
        """
        if simulation_func is None:
            # Use dummy function
            simulation_func = lambda x: {"dummy": 1.0}
        
        # Run samples
        sample_config = next(iter(self.generate_configs()))
        
        times = []
        for _ in range(sample_runs):
            start = time.time()
            simulation_func(sample_config)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        total_configs = len(self.generate_configs())
        total_runs = total_configs * self.n_replicates
        
        return {
            "avg_run_time": avg_time,
            "total_configs": total_configs,
            "total_runs": total_runs,
            "estimated_sequential_time": total_runs * avg_time,
            "estimated_parallel_time": total_runs * avg_time / self.n_workers,
        }


def sensitivity_analysis(
    sweep_result: SweepResult,
    output_metric: str
) -> Dict[str, float]:
    """
    Perform sensitivity analysis on sweep results.
    
    Calculates the relative importance of each parameter on the output metric.
    
    Args:
        sweep_result: Results from parameter sweep
        output_metric: Name of the metric to analyze
        
    Returns:
        Dictionary mapping parameter names to sensitivity scores
    """
    import pandas as pd
    from collections import defaultdict
    
    df = sweep_result.get_dataframe()
    
    if output_metric not in df.columns:
        # Try mean version
        output_metric = f"{output_metric}_mean"
        if output_metric not in df.columns:
            raise ValueError(f"Metric {output_metric} not found in results")
    
    sensitivities = {}
    
    for param in sweep_result.parameter_names:
        if param not in df.columns:
            continue
        
        # Group by parameter and calculate variance
        grouped = df.groupby(param)[output_metric].mean()
        
        # Calculate variance contribution
        overall_mean = df[output_metric].mean()
        param_variance = ((grouped - overall_mean) ** 2).sum()
        total_variance = df[output_metric].var()
        
        if total_variance > 0:
            sensitivities[param] = param_variance / total_variance
        else:
            sensitivities[param] = 0.0
    
    # Normalize to sum to 1
    total = sum(sensitivities.values())
    if total > 0:
        sensitivities = {k: v/total for k, v in sensitivities.items()}
    
    return dict(sorted(sensitivities.items(), key=lambda x: -x[1]))


def interaction_effects(
    sweep_result: SweepResult,
    output_metric: str
) -> Dict[Tuple[str, str], float]:
    """
    Calculate interaction effects between parameter pairs.
    
    Args:
        sweep_result: Results from parameter sweep
        output_metric: Name of the metric to analyze
        
    Returns:
        Dictionary mapping parameter pairs to interaction scores
    """
    import pandas as pd
    from itertools import combinations
    
    df = sweep_result.get_dataframe()
    metric_col = output_metric if output_metric in df.columns else f"{output_metric}_mean"
    
    interactions = {}
    params = sweep_result.parameter_names
    
    for p1, p2 in combinations(params, 2):
        # Calculate two-way grouping
        grouped = df.groupby([p1, p2])[metric_col].mean()
        
        # Calculate main effects
        main_p1 = df.groupby(p1)[metric_col].mean()
        main_p2 = df.groupby(p2)[metric_col].mean()
        overall = df[metric_col].mean()
        
        # Calculate expected values if no interaction
        interaction_score = 0.0
        for idx in grouped.index:
            expected = main_p1[idx[0]] + main_p2[idx[1]] - overall
            actual = grouped[idx]
            interaction_score += (actual - expected) ** 2
        
        interactions[(p1, p2)] = interaction_score
    
    return dict(sorted(interactions.items(), key=lambda x: -x[1]))
