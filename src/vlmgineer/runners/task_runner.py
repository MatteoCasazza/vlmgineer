"""
Task runner for executing a single task iteration with sampling, evaluation, and evolution.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

from vlmgineer.samplers.sampling_manager import SamplingManager
from vlmgineer.evaluator.evaluation_manager import EvaluationManager
from vlmgineer.evaluator.metrics import extract_run_metrics, RunMetrics


# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent.parent


def load_task_config(task_name: str, config_dir: Optional[str] = None) -> dict:
    """Load task-specific config from YAML file."""
    if config_dir is None:
        config_dir = PACKAGE_ROOT / "configs"
    config_path = Path(config_dir) / f"{task_name}.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(task_config: dict, overrides: dict) -> dict:
    """Merge task config with overrides (one level only)."""
    merged = task_config.copy()
    merged.update(overrides)
    return merged


def get_previous_designs(base_save_folder: str) -> list[str]:
    """Load previous tool designs from top_runs folder."""
    top_runs = []
    keys = ["strategy_name", "strategy_description", "tool_name", "tool_description", "tool_urdf"]
    top_runs_dir = os.path.join(base_save_folder, "top_runs")
    
    for folder in os.listdir(top_runs_dir):
        folder_path = os.path.join(top_runs_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                with open(os.path.join(folder_path, file), "r") as f:
                    run = json.load(f)
                    top_runs.append(json.dumps({k: run[k] for k in keys}, indent=4))
    return top_runs


@dataclass
class TaskRunResult:
    """Result from a single task run."""
    initial_metrics: RunMetrics
    final_metrics: RunMetrics
    all_runs: list[dict]
    base_save_folder: str
    
    @property
    def improvement(self) -> float:
        """Reward improvement from initial to final."""
        return self.final_metrics.reward - self.initial_metrics.reward


class TaskRunner:
    """
    Runs a single task iteration through sampling, evaluation, and evolution.
    
    For multiple runs, create a new TaskRunner instance or call run() multiple times
    from a loop in the calling code.
    """
    
    def __init__(
        self,
        task_name: str,
        config_overrides: Optional[dict] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize TaskRunner.
        
        Args:
            task_name: Name of the task to run
            config_overrides: Optional config overrides to apply
            log_fn: Optional logging function (defaults to print)
        """
        self.task_name = task_name
        self.log = log_fn or print
        
        # Load and merge config
        task_config = load_task_config(task_name)
        self.config = merge_configs(task_config, config_overrides or {})
    
    @property
    def top_k(self) -> int:
        return self.config.get("save_top_k", 5)
    
    @property
    def n_evolution(self) -> int:
        return self.config.get("n_evolution", 0)
    
    @property
    def stop_threshold(self) -> Optional[float]:
        return self.config.get("stop_reward_threshold")
    
    def run(self) -> TaskRunResult:
        """
        Run a single task iteration (initial sampling + evolution).
        
        Returns:
            TaskRunResult with initial/final metrics and all run data
        """
        self.log("=" * 80)
        self.log(f"Running task: {self.task_name}")
        self.log("=" * 80)
        
        all_runs = []
        
        # Initial sampling and evaluation
        sorted_runs, base_save_folder, run_buffer = self._run_sampling_and_evaluation()
        all_runs.extend(run_buffer)
        
        # Extract initial metrics
        initial_metrics = extract_run_metrics(sorted_runs, self.top_k)
        
        if not initial_metrics:
            raise RuntimeError(f"No valid runs produced for task {self.task_name}")
        
        self._log_metrics("Initial", initial_metrics)
        
        # Track best metrics (starts as initial)
        best_metrics = initial_metrics
        
        # Check if we need evolution
        if self.stop_threshold and initial_metrics.reward >= self.stop_threshold:
            self.log(f"Skipping evolution - initial reward {initial_metrics.reward:.4f} "
                    f"already exceeds threshold {self.stop_threshold:.4f}")
        elif self.n_evolution > 0:
            # Run evolution
            best_metrics, evo_runs = self._run_evolution(base_save_folder, initial_metrics)
            all_runs.extend(evo_runs)
        
        # Log final results
        self._log_metrics("Final", best_metrics)
        if best_metrics != initial_metrics:
            self.log(f"Improvement: {best_metrics.reward - initial_metrics.reward:.4f}")
        
        return TaskRunResult(
            initial_metrics=initial_metrics,
            final_metrics=best_metrics,
            all_runs=all_runs,
            base_save_folder=base_save_folder,
        )
    
    def _run_sampling_and_evaluation(self) -> tuple[list, str, list]:
        """Run sampling and evaluation, return sorted runs."""
        sm = SamplingManager(**self.config)
        base_save_folder = sm.begin_design_and_action_sampling()
        
        self.config["base_save_folder"] = base_save_folder
        em = EvaluationManager(**self.config)
        run_buffer = em.run(run_type="parallel")
        
        sorted_runs = sorted(run_buffer, key=lambda x: x['reward'], reverse=True)
        return sorted_runs, base_save_folder, run_buffer
    
    def _run_evolution(self, base_save_folder: str, initial_metrics: RunMetrics) -> tuple[RunMetrics, list]:
        """Run evolution iterations, return best metrics and all runs."""
        all_runs = []
        best_metrics = initial_metrics
        previous_designs = get_previous_designs(base_save_folder)
        
        for evo_idx in range(self.n_evolution):
            self.log(f"  Evolution {evo_idx + 1}/{self.n_evolution}")
            
            sm = SamplingManager(**self.config)
            base_save_folder = sm.evolve_design_and_action(previous_designs)
            
            self.config["base_save_folder"] = base_save_folder
            em = EvaluationManager(**self.config)
            run_buffer = em.run(run_type="parallel")
            all_runs.extend(run_buffer)
            
            sorted_evo_runs = sorted(run_buffer, key=lambda x: x['reward'], reverse=True)
            evo_metrics = extract_run_metrics(sorted_evo_runs, self.top_k)
            
            if evo_metrics and evo_metrics.reward > best_metrics.reward:
                best_metrics = evo_metrics
                self.log(f"    New best: {best_metrics.reward:.4f}")
            
            # Check early stopping
            if self.stop_threshold and best_metrics.reward >= self.stop_threshold:
                self.log(f"  Early stop - reached threshold {self.stop_threshold:.4f}")
                break
            
            previous_designs = get_previous_designs(base_save_folder)
        
        return best_metrics, all_runs
    
    def _log_metrics(self, label: str, metrics: RunMetrics) -> None:
        """Log metrics with a label."""
        k = self.top_k
        self.log(f"{label} results:")
        self.log(f"  Top reward: {metrics.reward:.4f}")
        self.log(f"  Top-{k} avg reward: {metrics.top_k_reward:.4f}")
        self.log(f"  Total distance: {metrics.total_dis:.4f}")
        self.log(f"  Top-{k} avg distance: {metrics.top_k_dis:.4f}")
        self.log(f"  Total time: {metrics.total_time:.4f}")
        self.log(f"  Top-{k} avg time: {metrics.top_k_time:.4f}")
