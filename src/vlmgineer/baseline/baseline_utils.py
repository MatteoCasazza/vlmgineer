import os
import json
from vlmgineer.samplers.sampling_agents import GeminiAgent, SamplingCollection
from vlmgineer.prompts.prompt_composer import SinglePrompt
from vlmgineer.evaluator.evaluation_manager import EvaluationManager
import prompts.prompt_utils as prompt_utils


def build_file_prompts(base_path, task_name, config):
    """
    Construct the list of file prompts for a given task.
    """
    file_prompts = []
    file_prompts += prompt_utils.get_universal_attachment_paths(base_path, config)
    file_prompts += prompt_utils.get_task_attachment_paths(base_path, task_name)
    if config.get("rlbench_tool") is not None:
        if config["rlbench_tool"]:
            tool_urdf_path = os.path.join(base_path, "models", f"{task_name}_tool", "tool.urdf")
            # convert URDF to txt and remove existing txt
            txt_path = tool_urdf_path.replace('.urdf', '.txt')
            if os.path.exists(txt_path):
                os.remove(txt_path)
            with open(tool_urdf_path, 'r') as f_in, open(txt_path, 'w') as f_out:
                f_out.write(f_in.read())
            file_prompts.append(txt_path)


    return file_prompts


def single_call_generic(run_idx, task_name, config, prompt_config, prompt_fn, schema, log_prefix, postprocess_fn=lambda x: x):
    """
    Generic function to send a prompt for a task and save JSON response.
    Returns the folder where samples are saved.
    """
    base_path = config.get("base_path", "")
    n_agent = config.get("n_agent", 1)
    model_name = config["model_name"]
    instruction = prompt_fn(prompt_config)
    file_prompts = build_file_prompts(base_path, task_name, config)
    prompts = [SinglePrompt(instruction_prompts=instruction, file_prompts=file_prompts) for _ in range(n_agent)]

    sc = SamplingCollection(n_agent, 50, model_name)

    gemini_cfg = {
        'response_mime_type': 'application/json',
        'response_schema': schema,
        'top_k': config['top_k'],
        'top_p': config['top_p'],
        'temperature': config['temperature'],
    }
    agent_configs = [gemini_cfg for _ in range(n_agent)]
    results = sc.start_sampling_agent(prompts, agent_configs)
    for i in range(len(results)):
        data = json.loads(results[i])
        data = postprocess_fn(data)

        if config.get("rlbench_tool") is not None:
            if config["rlbench_tool"]:
                data["tool_urdf"] = config["tool_urdf"]

        # split action_set into individual JSON files
        actions = data.pop('action_sets', [])
        print(f"Received {len(actions)} action sets for {task_name}")
        timestamp = config.get("timestamp")
        save_folder = os.path.join(log_prefix, timestamp, task_name, str(run_idx), "samples")
        os.makedirs(save_folder, exist_ok=True)

        for idx, action in enumerate(actions):
            item = data.copy()
            item['action_sets'] = [action]
            out_path = os.path.join(save_folder, f"{task_name}_{i}_{idx}.json")
            with open(out_path, 'w') as f:
                json.dump(item, f, indent=4)

        print(f"Saved {len(actions)} results for {task_name} to {save_folder}")
    return save_folder


def single_eval_generic(run_idx, task_name, config, log_prefix):
    """
    Generic evaluation for a task.
    """
    timestamp = config.get("timestamp")
    base_save_folder = os.path.join(log_prefix, timestamp, task_name, str(run_idx))
    config["base_save_folder"] = base_save_folder
    config["task_name"] = task_name
    em = EvaluationManager(**config)
    runs = em.run(run_type="parallel")
    sorted_runs = sorted(runs, key=lambda x: x['reward'], reverse=True)
    if not sorted_runs:
        return {"reward": None, "total_dis": None, "total_time": None}
    best = sorted_runs[0]
    # compute average metrics over all runs
    avg_reward = sum(r['reward'] for r in sorted_runs) / len(sorted_runs)
    avg_total_dis = sum(r['total_dis'] for r in sorted_runs) / len(sorted_runs)
    avg_total_time = sum(r['total_time'] for r in sorted_runs) / len(sorted_runs)
    return {
        "reward": best['reward'],
        "total_dis": best['total_dis'],
        "total_time": best['total_time'],
        # include averages
        "avg_reward": avg_reward,
        "avg_total_dis": avg_total_dis,
        "avg_total_time": avg_total_time
    }


def run_tasks(run_idx, task_list, config, prompt_fn, schema, log_prefix, postprocess_fn=lambda x: x, logger=print):
    """
    Run a sequence of tasks: generate samples and evaluate.
    Returns a dict of metrics per task.
    """

    os.makedirs(log_prefix, exist_ok=True)
    metrics = {}
    for i, task in enumerate(task_list):
        logger("="*80)
        logger(f"Start task: {task}")
        logger("="*80)

        if config.get("human_study") is not None:
            prompt_config = {'n_action': config['n_action'], 'human_prompts': config["human_prompts"][i]}
        else:
            prompt_config = {'n_action': config['n_action']}

        if config.get("rlbench_tool") is not None:
            if config["rlbench_tool"]:
                tool_urdf_path = os.path.join(config["base_path"], "models", f"{task}_tool", "tool.urdf")
                # convert URDF to txt and remove existing txt
                txt_path = tool_urdf_path.replace('.urdf', '.txt')
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                with open(tool_urdf_path, 'r') as urdf_file:
                    content = urdf_file.read()
                with open(txt_path, 'w') as txt_file:
                    txt_file.write(content)
                config["tool_urdf"] = content

        single_call_generic(run_idx, task, config, prompt_config, prompt_fn, schema, log_prefix, postprocess_fn)
        result = single_eval_generic(run_idx, task, config, log_prefix)
        metrics[task] = result
        logger(f"Completed {task}: best_reward={result['reward']}, best_reward_total_dis={result['total_dis']}, best_reward_total_time={result['total_time']}, avg_reward={result.get('avg_reward')}, avg_total_dis={result.get('avg_total_dis')}, avg_total_time={result.get('avg_total_time')}")
    return metrics
