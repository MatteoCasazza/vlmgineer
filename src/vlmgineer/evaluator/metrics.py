"""
Metrics tracking utilities for benchmark evaluation.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class RunMetrics:
    """Metrics for a single benchmark run."""
    reward: float = 0.0
    total_dis: float = 0.0
    total_time: float = 0.0
    top_k_reward: float = 0.0
    top_k_dis: float = 0.0
    top_k_time: float = 0.0


def extract_run_metrics(sorted_runs: list, top_k: int) -> Optional[RunMetrics]:
    """
    Extract all metrics from sorted runs.
    
    Args:
        sorted_runs: List of run results sorted by reward (descending)
        top_k: Number of top runs to average for top-k metrics
        
    Returns:
        RunMetrics object or None if sorted_runs is empty
    """
    if not sorted_runs:
        return None
    
    k = min(top_k, len(sorted_runs))
    top_k_runs = sorted_runs[:k]
    
    return RunMetrics(
        reward=sorted_runs[0]['reward'],
        total_dis=sorted_runs[0]['total_dis'],
        total_time=sorted_runs[0]['total_time'],
        top_k_reward=sum(r['reward'] for r in top_k_runs) / k,
        top_k_dis=sum(r['total_dis'] for r in top_k_runs) / k,
        top_k_time=sum(r['total_time'] for r in top_k_runs) / k,
    )


@dataclass
class TaskMetricsTracker:
    """
    Tracks metrics across multiple runs for a single task.
    
    Records initial (pre-evolution) and final (post-evolution) metrics,
    then computes aggregates like averages and bests.
    """
    top_k: int = 5
    
    # Per-run metrics lists
    initial_metrics: list[RunMetrics] = field(default_factory=list)
    final_metrics: list[RunMetrics] = field(default_factory=list)
    
    # Current run tracking (for evolution updates)
    _current_initial: Optional[RunMetrics] = field(default=None, repr=False)
    _current_best: Optional[RunMetrics] = field(default=None, repr=False)
    
    def record_initial(self, metrics: RunMetrics) -> None:
        """Record initial metrics for a run (before evolution)."""
        self.initial_metrics.append(metrics)
        self._current_initial = metrics
        # Initialize best with initial values
        self._current_best = RunMetrics(
            reward=metrics.reward,
            total_dis=metrics.total_dis,
            total_time=metrics.total_time,
            top_k_reward=metrics.top_k_reward,
            top_k_dis=metrics.top_k_dis,
            top_k_time=metrics.top_k_time,
        )
    
    def update_best(self, metrics: RunMetrics) -> None:
        """Update best metrics if evolution improved them."""
        if self._current_best is None:
            self._current_best = metrics
            return
        
        # Update best reward (higher is better)
        if metrics.reward > self._current_best.reward:
            self._current_best.reward = metrics.reward
            # When reward improves, also record associated dis/time
            self._current_best.total_dis = metrics.total_dis
            self._current_best.total_time = metrics.total_time
        
        # Update best top-k reward (higher is better)
        if metrics.top_k_reward > self._current_best.top_k_reward:
            self._current_best.top_k_reward = metrics.top_k_reward
            self._current_best.top_k_dis = metrics.top_k_dis
            self._current_best.top_k_time = metrics.top_k_time
    
    def record_final(self) -> None:
        """Record final metrics for current run (after all evolution)."""
        if self._current_best is not None:
            self.final_metrics.append(self._current_best)
        self._current_initial = None
        self._current_best = None
    
    @property
    def current_initial(self) -> Optional[RunMetrics]:
        """Get the initial metrics for the current run."""
        return self._current_initial
    
    @property
    def current_best(self) -> Optional[RunMetrics]:
        """Get the current best metrics (during evolution)."""
        return self._current_best
    
    # -------------------------------------------------------------------------
    # Aggregate computations
    # -------------------------------------------------------------------------
    
    def _safe_avg(self, values: list[float]) -> float:
        """Compute average, returning 0 if empty."""
        return sum(values) / len(values) if values else 0.0
    
    def _safe_max(self, values: list[float]) -> float:
        """Compute max, returning 0 if empty."""
        return max(values) if values else 0.0
    
    def _safe_min(self, values: list[float]) -> float:
        """Compute min, returning 0 if empty."""
        return min(values) if values else 0.0
    
    # Reward aggregates
    @property
    def avg_initial_reward(self) -> float:
        return self._safe_avg([m.reward for m in self.initial_metrics])
    
    @property
    def avg_final_reward(self) -> float:
        return self._safe_avg([m.reward for m in self.final_metrics])
    
    @property
    def best_initial_reward(self) -> float:
        return self._safe_max([m.reward for m in self.initial_metrics])
    
    @property
    def best_final_reward(self) -> float:
        return self._safe_max([m.reward for m in self.final_metrics])
    
    # Top-k reward aggregates
    @property
    def avg_initial_top_k_reward(self) -> float:
        return self._safe_avg([m.top_k_reward for m in self.initial_metrics])
    
    @property
    def avg_final_top_k_reward(self) -> float:
        return self._safe_avg([m.top_k_reward for m in self.final_metrics])
    
    # Distance aggregates
    @property
    def avg_initial_dis(self) -> float:
        return self._safe_avg([m.total_dis for m in self.initial_metrics])
    
    @property
    def avg_final_dis(self) -> float:
        return self._safe_avg([m.total_dis for m in self.final_metrics])
    
    @property
    def best_initial_dis(self) -> float:
        return self._safe_min([m.total_dis for m in self.initial_metrics])
    
    @property
    def best_final_dis(self) -> float:
        return self._safe_min([m.total_dis for m in self.final_metrics])
    
    # Top-k distance aggregates
    @property
    def avg_initial_top_k_dis(self) -> float:
        return self._safe_avg([m.top_k_dis for m in self.initial_metrics])
    
    @property
    def avg_final_top_k_dis(self) -> float:
        return self._safe_avg([m.top_k_dis for m in self.final_metrics])
    
    # Time aggregates
    @property
    def avg_initial_time(self) -> float:
        return self._safe_avg([m.total_time for m in self.initial_metrics])
    
    @property
    def avg_final_time(self) -> float:
        return self._safe_avg([m.total_time for m in self.final_metrics])
    
    @property
    def best_initial_time(self) -> float:
        return self._safe_min([m.total_time for m in self.initial_metrics])
    
    @property
    def best_final_time(self) -> float:
        return self._safe_min([m.total_time for m in self.final_metrics])
    
    # Top-k time aggregates
    @property
    def avg_initial_top_k_time(self) -> float:
        return self._safe_avg([m.top_k_time for m in self.initial_metrics])
    
    @property
    def avg_final_top_k_time(self) -> float:
        return self._safe_avg([m.top_k_time for m in self.final_metrics])
    
    # -------------------------------------------------------------------------
    # Improvements (computed properties)
    # -------------------------------------------------------------------------
    
    @property
    def avg_reward_improvement(self) -> float:
        return self.avg_final_reward - self.avg_initial_reward
    
    @property
    def best_reward_improvement(self) -> float:
        return self.best_final_reward - self.best_initial_reward
    
    @property
    def avg_top_k_reward_improvement(self) -> float:
        return self.avg_final_top_k_reward - self.avg_initial_top_k_reward
    
    @property
    def avg_dis_reduction(self) -> float:
        return self.avg_initial_dis - self.avg_final_dis
    
    @property
    def best_dis_reduction(self) -> float:
        return self.best_initial_dis - self.best_final_dis
    
    @property
    def avg_top_k_dis_reduction(self) -> float:
        return self.avg_initial_top_k_dis - self.avg_final_top_k_dis
    
    @property
    def avg_time_reduction(self) -> float:
        return self.avg_initial_time - self.avg_final_time
    
    @property
    def best_time_reduction(self) -> float:
        return self.best_initial_time - self.best_final_time
    
    @property
    def avg_top_k_time_reduction(self) -> float:
        return self.avg_initial_top_k_time - self.avg_final_top_k_time
    
    # -------------------------------------------------------------------------
    # Export methods
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> dict:
        """Export all metrics to a dictionary for JSON serialization."""
        return {
            # Raw per-run data
            'initial_top_rewards': [m.reward for m in self.initial_metrics],
            'final_top_rewards': [m.reward for m in self.final_metrics],
            'top_k_initial': [m.top_k_reward for m in self.initial_metrics],
            'top_k_final': [m.top_k_reward for m in self.final_metrics],
            
            # Reward aggregates
            'avg_initial': self.avg_initial_reward,
            'avg_final': self.avg_final_reward,
            'avg_improvement': self.avg_reward_improvement,
            'avg_top_k_initial': self.avg_initial_top_k_reward,
            'avg_top_k_final': self.avg_final_top_k_reward,
            'avg_top_k_improvement': self.avg_top_k_reward_improvement,
            'best_initial': self.best_initial_reward,
            'best_final': self.best_final_reward,
            'best_improvement': self.best_reward_improvement,
            
            # Distance data
            'initial_total_dis': [m.total_dis for m in self.initial_metrics],
            'final_total_dis': [m.total_dis for m in self.final_metrics],
            'avg_initial_dis': self.avg_initial_dis,
            'avg_final_dis': self.avg_final_dis,
            'avg_dis_reduction': self.avg_dis_reduction,
            'best_initial_dis': self.best_initial_dis,
            'best_final_dis': self.best_final_dis,
            'best_dis_reduction': self.best_dis_reduction,
            
            # Top-k distance data
            'top_k_dis_initial': [m.top_k_dis for m in self.initial_metrics],
            'top_k_dis_final': [m.top_k_dis for m in self.final_metrics],
            'avg_top_k_dis_initial': self.avg_initial_top_k_dis,
            'avg_top_k_dis_final': self.avg_final_top_k_dis,
            'avg_top_k_dis_reduction': self.avg_top_k_dis_reduction,
            'best_top_k_dis_initial': self._safe_min([m.top_k_dis for m in self.initial_metrics]),
            'best_top_k_dis_final': self._safe_min([m.top_k_dis for m in self.final_metrics]),
            'best_top_k_dis_reduction': self._safe_min([m.top_k_dis for m in self.initial_metrics]) - self._safe_min([m.top_k_dis for m in self.final_metrics]),
            
            # Time data
            'initial_total_time': [m.total_time for m in self.initial_metrics],
            'final_total_time': [m.total_time for m in self.final_metrics],
            'avg_initial_time': self.avg_initial_time,
            'avg_final_time': self.avg_final_time,
            'avg_time_reduction': self.avg_time_reduction,
            'best_initial_time': self.best_initial_time,
            'best_final_time': self.best_final_time,
            'best_time_reduction': self.best_time_reduction,
            
            # Top-k time data
            'top_k_time_initial': [m.top_k_time for m in self.initial_metrics],
            'top_k_time_final': [m.top_k_time for m in self.final_metrics],
            'avg_top_k_time_initial': self.avg_initial_top_k_time,
            'avg_top_k_time_final': self.avg_final_top_k_time,
            'avg_top_k_time_reduction': self.avg_top_k_time_reduction,
            'best_top_k_time_initial': self._safe_min([m.top_k_time for m in self.initial_metrics]),
            'best_top_k_time_final': self._safe_min([m.top_k_time for m in self.final_metrics]),
            'best_top_k_time_reduction': self._safe_min([m.top_k_time for m in self.initial_metrics]) - self._safe_min([m.top_k_time for m in self.final_metrics]),
        }
    
    def log_summary(self, log_fn: Callable[[str], None], task_name: str, total_runs: int) -> None:
        """Log a summary of task metrics."""
        k = self.top_k
        
        def pct(improvement: float, initial: float) -> str:
            """Format percentage improvement."""
            return f"{improvement / max(0.001, initial) * 100:.1f}%"
        
        log_fn("=" * 100)
        log_fn(f"Task {task_name} completed {total_runs} total runs")
        log_fn(f"Best initial reward: {self.best_initial_reward:.4f}")
        log_fn(f"Best final reward: {self.best_final_reward:.4f}")
        log_fn(f"Best reward improvement: {self.best_reward_improvement:.4f} ({pct(self.best_reward_improvement, self.best_initial_reward)})")
        log_fn("-" * 80)
        log_fn(f"Average initial top reward: {self.avg_initial_reward:.4f}")
        log_fn(f"Average final top reward: {self.avg_final_reward:.4f}")
        log_fn(f"Average top reward improvement: {self.avg_reward_improvement:.4f} ({pct(self.avg_reward_improvement, self.avg_initial_reward)})")
        log_fn("-" * 80)
        log_fn(f"Average initial top-{k} reward: {self.avg_initial_top_k_reward:.4f}")
        log_fn(f"Average final top-{k} reward: {self.avg_final_top_k_reward:.4f}")
        log_fn(f"Average top-{k} reward improvement: {self.avg_top_k_reward_improvement:.4f} ({pct(self.avg_top_k_reward_improvement, self.avg_initial_top_k_reward)})")
        log_fn(f"  Average initial top-{k} distance: {self.avg_initial_top_k_dis:.4f}")
        log_fn(f"  Average final   top-{k} distance: {self.avg_final_top_k_dis:.4f}")
        log_fn(f"  Reduction in avg top-{k} distance: {self.avg_top_k_dis_reduction:.4f}")
        log_fn(f"  Average initial top-{k} time: {self.avg_initial_top_k_time:.4f}")
        log_fn(f"  Average final   top-{k} time: {self.avg_final_top_k_time:.4f}")
        log_fn(f"  Reduction in avg top-{k} time: {self.avg_top_k_time_reduction:.4f}")
        log_fn("=" * 100)
