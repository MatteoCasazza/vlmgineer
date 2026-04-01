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
import vlmgineer.prompts.prompt_utils as prompt_utils
from datetime import datetime

from vlmgineer import envs
from vlmgineer.runners.env_runner import EnvRunner
import pybullet as p
import pybullet_utils.bullet_client as bc
from baseline.baseline_utils import run_tasks
import uuid

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rlbench_config = {
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

    "rlbench_tool": True,
    "n_parallel_rollout_processes": 50,
    "save_reward_threshold": 0.0,
}

def rlbench_prompt(prompt_config):
    n_action = prompt_config['n_action']
    return f"""
    You are a helpful robotics hardware and controls expert. You have a robot arm that sits in the origin of your environment. You are working with a colleague as a team to design tools and actions for a robot to complete a task. Your colleague will provide you with a design in the format of a URDF, which is attached for you as tool.txt. Your goal is to use your colleague's URDF to come up with an action plan for the robot to use.

    Here are some important specifications you need to follow for the output:

    (Action Specifications) Your tool-using action will be a Nx6 numpy array of action waypoints, where N is the number of waypoints, and each waypoint is of dimension 6 (xyz position + roll-pitch-yaw euler angle orientations). Your action needs to be precisely six numbers per waypoint. Your waypoints will be carried out by the EnvRunner class. It is important to stress this: the action waypoints are controlling the robot end-effector "panda_virtual" link: this means you have to carefully take into account the dimensions of the tool and the thickness of its parts when designing effective waypoints. Again, you can safely assume the end-effector has the same orientation as the world frame upon initialization (see frame clarification again for details)! Create {n_action} different action sets.

    (Frame Clarification) In the world frame, front/back is along the x axis, left/right is along the y axis and up/down is along the z axis with following directions: Positive x: Towards the front of the table. Negative x: Towards the back of the table. Positive y: Towards the left. Negative y: Towards the right. Positive z: Up, towards the ceiling. Negative z: Down, towards the floor. In terms of orientation, starting from the origin frame: Positive rotation about the x-axis: tilting end-effector head to the left. Negative rotation about the x-axis: tilting end-effector head to the right. Positive rotation about the y-axis: tilting end-effector head down. Negative rotation about the y-axis: tilting end-effector head up. Positive rotation about the z-axis: rotating the end-effector head counter-clockwise. Negative rotation about the z-axis: rotating the end-effector head clockwise. 

    (Units) All numbers described is in meters.

    (Output Format) For each strategy, you will output the following:
    - action_waypoints: a list of numpy arrays (containing {n_action} action sets)
    - action_description: a short description of the action you intend for these waypoints.

    """

def postprocess_no_tool(data):
    uid = uuid.uuid4().hex 
    
    data.update({
        "enable_gripper": False,
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
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to average")
    args = parser.parse_args()

    task_names = ["collect_and_elevate_spheres", "bring_cube_closer", "clean_table_top", "puck_to_goal"]
    accum = {t: {"reward":0, "total_dis":0, "total_time":0,
                "avg_reward":0, "avg_total_dis":0, "avg_total_time":0} for t in task_names}
    rlbench_config["base_path"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_prefix = os.path.join("baseline", "logs_rlbench")
    log_dir = os.path.join(log_prefix, timestamp)
    logger = Logger(log_dir)
    log = logger.log

    for run_idx in range(args.num_runs):
        log(f"\n=== Run {run_idx}/{args.num_runs} ===")
        metrics = run_tasks(run_idx, task_names, rlbench_config, rlbench_prompt, ActionResponseSchema, log_prefix, postprocess_fn=postprocess_no_tool, logger=log)
        for t, m in metrics.items():
            accum[t]["reward"]        += m.get("reward", 0)
            accum[t]["total_dis"]     += m.get("total_dis", 0)
            accum[t]["total_time"]    += m.get("total_time", 0)
            accum[t]["avg_reward"]    += m.get("avg_reward", 0)
            accum[t]["avg_total_dis"] += m.get("avg_total_dis", 0)
            accum[t]["avg_total_time"]+= m.get("avg_total_time", 0)

    log("\n=== Averages over runs ===")
    for t in task_names:
        avg_r   = accum[t]["reward"]        / args.num_runs
        avg_d   = accum[t]["total_dis"]      / args.num_runs
        avg_tm  = accum[t]["total_time"]     / args.num_runs
        avg_ar  = accum[t]["avg_reward"]     / args.num_runs
        avg_ad  = accum[t]["avg_total_dis"]   / args.num_runs
        avg_at  = accum[t]["avg_total_time"] / args.num_runs
        log(f"{t}: avg_best_reward={avg_r:.3f}, avg_best_reward_distance={avg_d:.3f}, avg_best_reward_time={avg_tm:.3f}, avg_mean_reward={avg_ar:.3f}, avg_mean_distance={avg_ad:.3f}, avg_mean_time={avg_at:.3f}")

if __name__ == "__main__":
    main()
    # import baseline_utils
    # rlbench_config["timestamp"] = "20250510_151229"
    # baseline_utils.single_eval_generic(0, "rlbench_hockey", rlbench_config, "baseline/logs_rlbench")
