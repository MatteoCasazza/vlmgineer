import numpy as np
from vlmgineer import envs 
from vlmgineer.runners.env_runner import EnvRunner
import pybullet as p
import pybullet_utils.bullet_client as bc

# task_name = 'bring_cube_closer'  # Change this to the desired task name
task_names = {
    "bring_cube_closer": "bring_cube_closer",
    # "dislodge_cube": "dislodge_cube",
    # "clean_table_top": "clean_table_top",
    # "puck_to_goal": "puck_to_goal",
    # "retrieve_high_object": "retrieve_high_object",
    # "serve_turkey_legs": "serve_turkey_legs", 
    # "bring_screen_down": "bring_screen_down",
    # "take_one_book_out": "take_one_book_out",
    # "rlbench_hockey": "rlbench_hockey",
    # "rlbench_scoop_with_spatula": "rlbench_scoop_with_spatula",
    # "rlbench_sweep_to_dustpan": "rlbench_sweep_to_dustpan",
    # "rlbench_reach_and_drag": "rlbench_reach_and_drag",
    # "collect_and_elevate_spheres": "collect_and_elevate_spheres",
    # "move_ball": "move_ball",
    # "place_cubes_inside": "place_cubes_inside",
    # "gather_strawberries": "gather_strawberries",
    # "close_plier": "close_plier",
    # "lift_cookie_jar": "lift_cookie_jar",
}

waypoints = np.array([
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.0, 0.4, 0.0, 0.1, 0.0, 0.0],
    [0.4, 0.0, 0.4, 0.0, 0.2, 0.0, 1.0],
    [0.4, 0.0, 0.4, 0.0, 0.3, 0.0, 1.0],
    [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.5, 0.0, 1.0],
    # [0.4, 0.0, 0.4, 0.0, 0.6, 0.0, 1.0],
])

physics_client = bc.BulletClient(connection_mode=p.GUI)

for task_name in task_names:
    env = envs.env_dict[task_name](
        robot_urdf_path="panda/panda.urdf", 
        physics_client=physics_client,
        # enable_gripper=True,
        screenshot_save_path=f'envs/{task_name}'
    )
    runner = EnvRunner(
        env=env, 
        waypoints=waypoints, 
        enable_orn=True,
        # enable_gripper=True,
    )
    print(f"Reward: {runner.run()}")