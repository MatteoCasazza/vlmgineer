from typing import List, Dict, Any, Optional
import numpy as np

import vlmgineer.envs as envs
from vlmgineer.runners.env_runner import EnvRunner

class BaseRunManager:
    def __init__(self, 
                 task_name: str, 
                 env_args: List[Dict[str, Any]], 
                 action_sets: Any, 
        ):
        """
        Base run manager initialization.
        
        Args:
            task_name: Name of the task to run
            env_args: List of environment arguments for each run
            action_sets: List of action sets for each run
        """
        self.task_name = task_name
        self.env_args = env_args
        self.action_sets = action_sets
        self.results = []
        
        assert len(env_args) == len(action_sets), "Environment args and action sets must have the same length"
    
    def _create_env(self, env_arg: Dict[str, Any]):
        """Create an environment instance with the given arguments."""
        env_class = envs.get_env_class(self.task_name)
        return env_class(**env_arg)
    
