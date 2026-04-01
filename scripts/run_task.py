"""
Run a single task with sampling, evaluation, and optional evolution.
"""
import click
from vlmgineer.runners.task_runner import TaskRunner


@click.command()
@click.option('--task', default='bring_cube_closer', help='Name of task to run')
def main(task: str):
    """Run a single task iteration."""
    runner = TaskRunner(task_name=task)
    result = runner.run()
    
    print(f"\n{'='*40}")
    print(f"Task: {task}")
    print(f"  Initial reward: {result.initial_metrics.reward:.4f}")
    print(f"  Final reward:   {result.final_metrics.reward:.4f}")
    print(f"  Improvement:    {result.improvement:.4f}")
    print(f"  Save folder:    {result.base_save_folder}")


if __name__ == "__main__":
    main()
