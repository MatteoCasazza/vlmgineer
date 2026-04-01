import os
import multiprocessing as mp
from multiprocessing import Process, Pipe
from typing import List, Dict, Any
import copy
import time

from vlmgineer.runners.base_run_manager import BaseRunManager
from vlmgineer.runners.env_runner import EnvRunner

# Initialize multiprocessing support
mp.freeze_support()

# Message commands
_RUN = 1
_CLOSE = 2
_TIMEOUT = 240  # Timeout in seconds for worker processes

def worker_process(rank: int, num_processes: int, child_pipe, task_name: str, env_arg: Dict[str, Any], actions: Any):
    """Worker process that runs a single environment simulation"""
    print(f"Worker {rank} of {num_processes} started")
    
    try:
        # Create environment
        import vlmgineer.envs as envs
        env_class = envs.get_env_class(task_name)
        
        # Ensure GUI is disabled for workers
        env_arg['enable_gui'] = False
        if 'physics_client' in env_arg:
            del env_arg['physics_client']
            
        env = env_class(**env_arg)
        runner = EnvRunner(env, actions, enable_gripper=env_arg['enable_gripper'])
        
        while True:
            try:
                if not child_pipe.poll(0.0001):
                    continue
                message, _ = child_pipe.recv()
            except (EOFError, KeyboardInterrupt):
                break
                
            if message == _RUN:
                # Run simulation and send results back with rank to ensure ordering
                result = runner.run()
                child_pipe.send([rank, result])
                continue
                
            elif message == _CLOSE:
                if hasattr(env, 'close'):
                    env.close()
                child_pipe.send(["close ok"])
                break
                
    except Exception as e:
        child_pipe.send([rank, {'error': str(e)}])
    finally:
        child_pipe.close()


class ParallelRunManager(BaseRunManager):
    def __init__(self, 
                 task_name: str, 
                 env_args: List[Dict[str, Any]], 
                 action_sets: Any, 
                 max_workers: int = 10,
                 timeout: int = _TIMEOUT
        ):
        """Initialize ParallelRunManager for running environments in parallel"""
        super().__init__(task_name, env_args, action_sets)
        self.max_workers = min(max_workers, len(env_args))
        self.timeout = timeout
    
    def run(self):
        """Run all environment-action pairs in parallel and return results."""
        print(f"Running {self.task_name} with {len(self.env_args)} tasks (max {self.max_workers} at once)")

        # Deep copy args for processes
        env_args_copies = [copy.deepcopy(arg) for arg in self.env_args]
        action_copies = [copy.deepcopy(actions) for actions in self.action_sets]
        
        num_tasks = len(self.env_args)
        results = [None] * num_tasks
        
        for batch_start in range(0, num_tasks, self.max_workers):
            batch_end = min(batch_start + self.max_workers, num_tasks)
            batch_size = batch_end - batch_start
            
            # Create processes and pipes
            processes = []
            child_pipes = []
            parent_pipes = []
            
            # Set up pipes for communication
            for i in range(batch_size):
                parent_pipe, child_pipe = Pipe()
                parent_pipes.append(parent_pipe)
                child_pipes.append(child_pipe)
            
            # Start worker processes
            for i in range(batch_size):
                rank = batch_start + i
                p = mp.Process(
                    target=worker_process, 
                    args=(rank, num_tasks, child_pipes[i], self.task_name, 
                          env_args_copies[rank], action_copies[rank])
                )
                p.start()
                processes.append(p)
            
            # Send run command to all processes
            for parent_pipe in parent_pipes:
                parent_pipe.send([_RUN, None])
            
            # Collect results with timeout
            remaining = list(range(batch_size))
            start_time = time.time()
            
            while remaining and time.time() - start_time < self.timeout:
                for i in remaining[:]:
                    if parent_pipes[i].poll(0.01):
                        worker_rank, result = parent_pipes[i].recv()
                        # Store result at the correct position based on worker_rank
                        results[worker_rank] = result
                        print(f"Worker {worker_rank} finished")
                        remaining.remove(i)
                
                if remaining:
                    time.sleep(0.01)
            
            # Handle timeouts
            for i in remaining:
                worker_rank = batch_start + i
                results[worker_rank] = {'error': f'Process timed out after {self.timeout} seconds'}
            
            # Send close command to all processes
            for parent_pipe in parent_pipes:
                parent_pipe.send([_CLOSE, None])
            
            # Join processes and close pipes
            for p in processes:
                p.join(1.0)  # Short timeout to avoid hanging
                if p.is_alive():
                    p.terminate()
            
            for pipe in parent_pipes:
                pipe.close()
        
        self.results = results
        return results