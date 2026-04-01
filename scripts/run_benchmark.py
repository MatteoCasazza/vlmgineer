"""
Benchmark runner for evaluating multiple tasks with multiple runs.
"""
import os
import json
from dataclasses import dataclass, field

from vlmgineer.common.utils import Logger, create_timestamped_log_dir
from vlmgineer.runners.task_runner import TaskRunner, TaskRunResult


@dataclass
class BenchmarkResults:
    """Aggregated results across multiple runs."""
    task_name: str
    results: list[TaskRunResult] = field(default_factory=list)
    
    @property
    def n_runs(self) -> int:
        return len(self.results)
    
    @property
    def best_reward(self) -> float:
        if not self.results:
            return 0.0
        return max(r.final_metrics.reward for r in self.results)
    
    @property
    def avg_reward(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.final_metrics.reward for r in self.results) / len(self.results)
    
    @property
    def avg_improvement(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.improvement for r in self.results) / len(self.results)
    
    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "n_runs": self.n_runs,
            "best_reward": self.best_reward,
            "avg_reward": self.avg_reward,
            "avg_improvement": self.avg_improvement,
            "runs": [
                {
                    "initial_reward": r.initial_metrics.reward,
                    "final_reward": r.final_metrics.reward,
                    "improvement": r.improvement,
                    "base_save_folder": r.base_save_folder,
                }
                for r in self.results
            ]
        }


# --- Benchmark configuration ---
benchmark_config = {
    "tasks": [
        "elevate_plate",
        "bring_cube_closer",
        "clean_table_top",
        # "collect_and_elevate_spheres",
        # "dislodge_cube",
        # "serve_turkey_legs",
        # "take_cookie_out_from_jar",
        # "move_ball",
        # "puck_to_goal",
        # "retrieve_high_object",
        # "take_one_book_out",
    ],
    "n_runs_per_task": 1,
    "overrides": {
        "n_agent": 20,
        "n_tool_samples_batch_size": 10,
        "n_action_samples_batch_size": 10,
        "model_name": "gemini-2.5-pro",
        "n_parallel_rollout_processes": 20,
        "stop_reward_threshold": float('inf'),
        "n_evolution": 3,
    }
}


def main():
    """Run benchmark across all configured tasks."""
    # Setup logging
    log_dir, timestamp = create_timestamped_log_dir(
        os.path.join(os.getcwd(), 'logs'), 
        prefix='benchmark'
    )
    logger = Logger(log_dir)
    log = logger.log
    
    log(f"Starting benchmark with {len(benchmark_config['tasks'])} tasks")
    log(f"Runs per task: {benchmark_config['n_runs_per_task']}")
    log(f"Log directory: {log_dir}")
    
    all_results = {}
    
    for task_name in benchmark_config["tasks"]:
        log(f"\n{'='*80}")
        log(f"TASK: {task_name}")
        log(f"{'='*80}")
        
        task_results = BenchmarkResults(task_name=task_name)
        
        for run_idx in range(benchmark_config["n_runs_per_task"]):
            log(f"\n--- Run {run_idx + 1}/{benchmark_config['n_runs_per_task']} ---")
            
            runner = TaskRunner(
                task_name=task_name,
                config_overrides=benchmark_config["overrides"],
                log_fn=log,
            )
            
            result = runner.run()
            task_results.results.append(result)
            
            log(f"Run {run_idx + 1} complete: reward={result.final_metrics.reward:.4f}")
        
        # Log task summary
        log(f"\nTask {task_name} summary:")
        log(f"  Best reward: {task_results.best_reward:.4f}")
        log(f"  Avg reward:  {task_results.avg_reward:.4f}")
        log(f"  Avg improvement: {task_results.avg_improvement:.4f}")
        
        all_results[task_name] = task_results.to_dict()
    
    # Write all results to JSON
    results_file = os.path.join(log_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to: {results_file}")
    
    # Print final summary
    log(f"\n{'='*80}")
    log("BENCHMARK COMPLETE")
    log(f"{'='*80}")
    for task_name, results in all_results.items():
        log(f"{task_name}: best={results['best_reward']:.4f}, avg={results['avg_reward']:.4f}")


if __name__ == "__main__":
    main()
