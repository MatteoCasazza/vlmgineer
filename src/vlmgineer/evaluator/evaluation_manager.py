import numpy as np
import os
import glob
import re
import json
import pybullet as p
import pybullet_utils.bullet_client as bc
import vlmgineer.envs as envs

from vlmgineer.runners import ParallelRunManager
from vlmgineer.runners.env_runner import EnvRunner
from vlmgineer.common.utils import slugify



class EvaluationManager():
    def __init__(self, **kwargs):
        self.task_name = kwargs.get('task_name')
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_save_folder = kwargs.get('base_save_folder')
        self.save_top_k = kwargs.get('save_top_k', 5)
        self.save_reward_threshold = kwargs.get('save_reward_threshold', 0.5)
        self.check_self_collision = kwargs.get('check_self_collision', False)
        self.n_parallel_rollout_processes = kwargs.get('n_parallel_rollout_processes', 10)
        self.run_buffer = []

        self._clean_files()
        self._load_run_buffer()

    def _clean_files(self):
        # Remove any panda files left
        for f in glob.glob(os.path.join(self.base_path, "models", "panda", "panda_*.urdf")): os.remove(f)

    def _load_run_buffer(self):
        sample_save_folder = os.path.join(self.base_save_folder, "samples")
        global_tool_id = 0
        for file_path in os.listdir(sample_save_folder):
            try:
                with open(os.path.join(sample_save_folder, file_path), 'r') as f:
                    strategy = json.load(f)
                if strategy['enable_gripper']:
                    robot_read_path = os.path.join(self.base_path, "prompts", "robots", "pandaFingers.txt")
                else:
                    robot_read_path = os.path.join(self.base_path, "prompts", "robots", "panda.txt")
                robot_local_path = os.path.join("panda", f"panda_tool_{global_tool_id}.urdf")
                robot_write_path = os.path.join(self.base_path, "models", robot_local_path)
                robot_tool_urdf = self._combine_tool_robot_urdf(strategy['tool_urdf'], robot_read_path, robot_write_path)

                for action_set in strategy['action_sets']:
                    self.run_buffer.append(self._get_run_entry(
                        env_arg={
                            'robot_urdf_path': robot_local_path,
                            'enable_gripper': strategy['enable_gripper'],
                            'check_self_collision': self.check_self_collision,
                        },
                        action_set=action_set,
                        task_name=self.task_name,
                        strategy_description=strategy["strategy_description"],
                        strategy_name=slugify(strategy["strategy_name"]),
                        tool_id=global_tool_id,
                        tool_name=slugify(strategy["tool_name"]),
                        tool_description=strategy["tool_description"],
                        full_urdf=robot_tool_urdf,
                        tool_urdf=strategy['tool_urdf'],
                    ))
                # increment after all action associated is added into run history
                global_tool_id += 1
            except Exception as e:
                print(f"Error loading sample file {file_path}: {str(e)}. Skipping...")
                continue

    def _combine_tool_robot_urdf(self, raw_tool_urdf, robot_read_path, robot_write_path):
        # Parse urdf
        xml_str = re.sub(r'<\?xml[^>]*\?>\s*', '', raw_tool_urdf)
        xml_str = re.sub(r'<robot[^>]*>\s*', '', xml_str)
        xml_str = re.sub(r'\s*</robot>\s*$', '', xml_str)

        with open(robot_read_path, "r") as f:
            base_robot_file = f.read()
        
        # Insert the XML content within <robot name="panda" ...></robot> tags
        robot_tool_urdf = re.sub(
            r'(<robot name="panda" xmlns:xacro="http://www.ros.org/wiki/xacro">.*?)(</robot>)',
            r"\1\n" + xml_str + r"\n\2",
            base_robot_file,
            flags=re.DOTALL
        )

        with open(robot_write_path, "w") as f:
            f.write(robot_tool_urdf)
        
        return robot_tool_urdf

    def _get_run_entry(self, **kwargs):
        return kwargs

    def run(self, run_type="parallel"):
        """Run all environment-action pairs and return results."""
        env_args = [entry['env_arg'] for entry in self.run_buffer]
        action_sets = [entry['action_set'] for entry in self.run_buffer]

        while True:
            try:

                manager = ParallelRunManager(
                    task_name=self.task_name,
                    env_args=env_args,
                    action_sets=action_sets,
                    max_workers=self.n_parallel_rollout_processes,
                )

                results = manager.run()
                break

            except Exception as e:
                print(f"Error running rollouts: {e}. \nRestarting")
                continue
        
        # Update run_buffer with results
        for run_idx, result in enumerate(results):
            if 'error' in result:
                print(f"Error in evaluating run {run_idx}: {result['error']}. Discarding run.")
                self.run_buffer[run_idx] = None
            else:
                self.run_buffer[run_idx] = {
                    **self.run_buffer[run_idx],
                    **results[run_idx]
                }
        
        self.run_buffer = list(filter(lambda x: x is not None, self.run_buffer))
        
        self._save_run_results()
        self._save_top_k()
        
        return self.run_buffer
        

    def _save_top_k(self):
        """
        Save the top k runs based on reward.
        Only include one run per tool (the one with the highest reward).
        Only saves runs with reward greater than or equal to the threshold.
        Handles errors by skipping problematic runs while maintaining k top runs.
        """
        # Sort all runs by reward in descending order
        sorted_runs = sorted(self.run_buffer, key=lambda x: x['reward'], reverse=True)
        seen_tool_ids = set()
        final_runs = []
        run_idx = 0

        physics_client = bc.BulletClient(p.GUI)
        top_run_save_folder = os.path.join(self.base_save_folder, "top_runs")
        os.makedirs(top_run_save_folder, exist_ok=True)

        try:
            while len(final_runs) < self.save_top_k and run_idx < len(sorted_runs):
                run = sorted_runs[run_idx]
                run_idx += 1

                # Skip if reward is below threshold or tool already seen
                if run['reward'] < self.save_reward_threshold or run['tool_id'] in seen_tool_ids:
                    continue
                print(len(final_runs))
                print(run['reward'])
                try:
                    strategy_name = run['strategy_name']
                    tool_id = run['tool_id']
                    run_folder_name = f"rank_{len(final_runs)+1}_tool_{tool_id}_{strategy_name}_reward_{run['reward']:.4f}"
                    top_run_folder = os.path.join(top_run_save_folder, run_folder_name)
                    os.makedirs(top_run_folder, exist_ok=True)

                    # Re-simulate with visualization
                    env = envs.get_env_class(self.task_name)(
                        physics_client=physics_client,
                        video_save_path=os.path.join(top_run_folder, f"rollout.mp4"),
                        blender_recorder=None,
                        **run['env_arg'],
                    )
                    waypoints = np.array(run['action_set'])
                    runner = EnvRunner(env=env, waypoints=waypoints, enable_gripper=run['env_arg']['enable_gripper'])
                    runner.run()

                    # Save run data
                    run_file = os.path.join(top_run_folder, "run.json")
                    with open(run_file, "w") as f:
                        json.dump(run, f, indent=4)

                    final_runs.append(run)
                    seen_tool_ids.add(tool_id)
                except Exception as e:
                    print(f"Error recreating top run: {str(e)}. Skipping...")
                    continue
        finally:
            physics_client.disconnect()

        print(f"Successfully saved {len(final_runs)} runs to {top_run_save_folder}")
        return top_run_save_folder

    def _save_run_results(self):
        run_buffer_save_folder = os.path.join(self.base_save_folder, "all_runs")
        os.makedirs(run_buffer_save_folder, exist_ok=True)
        run_buffer_file = os.path.join(run_buffer_save_folder, f"run_buffer.json")
        with open(run_buffer_file, 'w') as f:
            json.dump(self.run_buffer, f, indent=4)
        print(f"Run buffer saved in {run_buffer_save_folder}")
        return run_buffer_save_folder

