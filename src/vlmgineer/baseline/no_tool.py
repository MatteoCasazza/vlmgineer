import os
import argparse
import json
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlmgineer.common.utils import Logger
from vlmgineer.samplers.sampling_agents import GeminiAgent
from vlmgineer.prompts.prompt_composer import SinglePrompt
from vlmgineer.evaluator.evaluation_manager import EvaluationManager
from vlmgineer.prompts.schemas.response_schema import ActionResponseSchema
import prompts.prompt_utils as prompt_utils
from datetime import datetime

import envs
from vlmgineer.runners.env_runner import EnvRunner
import pybullet as p
import pybullet_utils.bullet_client as bc
from baseline.baseline_utils import run_tasks
from vlmgineer.prompts.schemas.response_schema import ActionResponseSchema
import uuid

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
notool_config = {
    # benchmarking parameters
    "base_path": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "timestamp": timestamp,
    "n_agent" : 5,
    "n_action": 10,

    # sampling parameters
    "model_name": "gemini-2.5-pro-preview-03-25",
    "top_k": 40,
    "top_p": 0.95,
    "temperature": 1.0,
    "max_thinking": True,

    "save_top_k": 2,
    "n_parallel_rollout_processes": 70,
    "save_reward_threshold": -0.01,
}

def notool_prompt(prompt_config):
    n_action = prompt_config['n_action']
    return f"""
    You are a robotics hardware and controls expert. You operate with boldness and brilliance in the physical realm. You work with a robot arm that sits in the origin of your environment. You will be presented with some robotic task, and will be asked to design actions to complete the task.

    The procedure you will follow:
        1. Receive Environment Descriptions: The user will provide some detailed environment description, robotic task instruction, and an initial image of the workspace area from the camera.
        2. Describe the Scene: Analyze the environment. Write down the spatial relationship, including by not limited to the position, orientation, dimension, and geometry of all the objects in the scene. Use all information provided to you, including all text, code, and image.
        3. Create Actions: You will need to generate {n_action} set of action waypoints that you can use to complete the task. 
            (f) Use your in-depth analysis regarding the intricate 3D spatial relationships within the environment, create {n_action} number of different step by step action plans. Be very wary about how objects interacts with each other!
            (g) Transform your step-by-step action plan into waypoints adhering to the "Action Specifications". During this transformation, think about the inherent nature of controlling robots with waypoint control and the difficulty that may present.

    (Desired Action Criteria Definitions) For the description below, we will call a single sequential set of waypoints in a single rollout as one "action set". For each tool you created, the goal is to generate {n_action} action sets that optimizes the task-successfulness and motion differentiation. Task-successfulness is optimized when an action set is able to complete the task successfully. Motion differentiation is optimized when there exists a large variance in the motion taken across all action sets you design for the same tool. A large variance in motion is defined the tool, at each time step, is located at a different location in the 3D space.Think about how a tool can be used to interact with the object from many different sides, angles, and ways. When both conditions are met, you have successfully designed a good set of action sets.

    (Action Specifications) Your tool-using action will be a Nx7 numpy array of action waypoints, where N is the number of waypoints, and each waypoint is of dimension 7 (xyz position + roll-pitch-yaw euler angle orientations + binary gripper open/close state in integers [0 for open, 1 for close]). Your action needs to be precisely seven numbers per waypoint. Your waypoints will be carried out by the EnvRunner class. It is important to stress this: the action waypoints are controlling the robot end-effector "panda_virtual" link: this means you have to carefully take into account the dimensions of the tool and the thickness of its parts when designing effective waypoints. Again, you can safely assume the end-effector has the same orientation as the world frame upon initialization (see frame clarification again for details)!

    (Frame Clarification) In the world frame, front/back is along the x axis, left/right is along the y axis and up/down is along the z axis with following directions: Positive x: Towards the front of the table. Negative x: Towards the back of the table. Positive y: Towards the left. Negative y: Towards the right. Positive z: Up, towards the ceiling. Negative z: Down, towards the floor. 

    (Robot Workspace Specification) The robot workspace is a semi-circle with the following dimensions: 
            - The center of the semi-circle is at the origin of the robot frame.
            - The radius of the semi-circle is 0.8 meters.
            - It points towards the positive z direction, meaning it spans from the positive to negative x and y axis, and only toward the positive z direction.
            This workspace constrain should make you think about which regions in the environment are accessible to the robot, and which are not. Your tool should be there to help the robot interact with those regions that are not accessible!

    (Units) All numbers described is in meters.

    (Output Format) For each strategy, you will output the following:
    - action_waypoints: a list of numpy arrays (containing {n_action} action sets)
    - action_description: a short description of the action you intend for these waypoints.

    """

def postprocess_no_tool(data):
    uid = uuid.uuid4().hex 
    data.update({
        "enable_gripper": True,
        "tool_urdf": "",
        "strategy_name": f"{uid}",
        "strategy_description": "N/A",
        "tool_id": 1,
        "tool_name": f"fname_{uid}",
        "tool_description": ""
    })
    return data

def main():
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to average")
    args = parser.parse_args()

    # task_names = ["bring_cube_closer", "clean_table_top", "collect_and_elevate_spheres", "dislodge_cube", "elevate_plate", "move_ball", "puck_to_goal", "retrieve_high_object", "serve_turkey_legs", "take_one_book_out"]
    task_names = ["bring_cube_closer", "clean_table_top", "collect_and_elevate_spheres", "dislodge_cube", "elevate_plate"]
    accum = {t: {"reward":0, "total_dis":0, "total_time":0,
                "avg_reward":0, "avg_total_dis":0, "avg_total_time":0} for t in task_names}
    best = {t: {"best_reward": float("-inf"),
               "total_dis_at_best": 0.0,
               "total_time_at_best": 0.0} for t in task_names}
    notool_config["base_path"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_prefix = os.path.join("baseline", "logs_notool")
    log_dir = os.path.join(log_prefix, timestamp)
    logger = Logger(log_dir)
    log = logger.log

    for run_idx in range(args.num_runs):
        log(f"\n=== Run {run_idx+1}/{args.num_runs} ===")
        metrics = run_tasks(run_idx, task_names, notool_config, notool_prompt, ActionResponseSchema, log_prefix, postprocess_fn=postprocess_no_tool, logger=log)
        for t, res in metrics.items():
            accum[t]['reward']        += res.get('reward', 0)
            accum[t]['total_dis']     += res.get('total_dis', 0)
            accum[t]['total_time']    += res.get('total_time', 0)
            accum[t]['avg_reward']    += res.get('avg_reward', 0)
            accum[t]['avg_total_dis'] += res.get('avg_total_dis', 0)
            accum[t]['avg_total_time']+= res.get('avg_total_time', 0)
            # update best run based on reward
            if res.get("reward", 0) > best[t]["best_reward"]:
                best[t]["best_reward"]        = res.get("reward", 0)
                best[t]["total_dis_at_best"]  = res.get("total_dis", 0)
                best[t]["total_time_at_best"] = res.get("total_time", 0)

    log("\n=== Averages over runs ===")
    for t in task_names:
        avg_r  = accum[t]['reward']        / args.num_runs
        avg_d  = accum[t]['total_dis']      / args.num_runs
        avg_tm = accum[t]['total_time']     / args.num_runs
        avg_ar = accum[t]['avg_reward']     / args.num_runs
        avg_ad = accum[t]['avg_total_dis']   / args.num_runs
        avg_at = accum[t]['avg_total_time'] / args.num_runs
        log(f"{t}: avg_best_reward={avg_r:.3f}, avg_best_distance={avg_d:.3f}, avg_best_time={avg_tm:.3f}, "
            f"avg_mean_reward={avg_ar:.3f}, avg_mean_distance={avg_ad:.3f}, avg_mean_time={avg_at:.3f}")

    # print best metrics
    log("\n=== Bests over runs ===")
    for t in task_names:
        br = best[t]["best_reward"]
        bd = best[t]["total_dis_at_best"]
        bt = best[t]["total_time_at_best"]
        log(f"{t}: best_reward={br:.3f}, total_distance_at_best_reward={bd:.3f}, total_time_at_best_reward={bt:.3f}")

if __name__ == '__main__':
    main()
