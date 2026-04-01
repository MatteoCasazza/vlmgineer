import os
import glob
import re
import json
import random
import numpy as np
import pybullet as p
import time  # Import time for retries
from typing import List, Dict, Any, Optional
from datetime import datetime

from google import genai
from google.genai import types

import vlmgineer.envs as envs 
from vlmgineer.prompts.schemas.response_schema import InitialResponseSchema
from vlmgineer.prompts.prompt_composer import PromptComposer
from vlmgineer.prompts.prompt_utils import get_universal_attachment_paths, get_task_attachment_paths
from vlmgineer.prompts.prompt_composer import SinglePrompt
from vlmgineer.samplers.sampling_agents import SamplingCollection
from vlmgineer.common.utils import slugify


class SamplingManager():
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', 'gemini-2.5-pro-preview-03-25')
        self.task_name = kwargs.get('task_name')
        self.max_n_query_threads = kwargs.get('max_n_query_threads', 50)
        self.n_agent = kwargs.get('n_agent', 30)
        self.n_tool_samples_batch_size = kwargs.get('n_tool_samples_batch_size', 5)
        self.n_action_samples_batch_size = kwargs.get('n_action_samples_batch_size', 5)
        self.temperature = kwargs.get('temperature', 1.0)
        self.random_temperature = kwargs.get('random_temperature', True)
        self.enable_gripper = kwargs.get('enable_gripper', False)
        self.random_gripper = kwargs.get('random_gripper', True)
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prompt_composer = PromptComposer(**kwargs)
        self._make_config_list()

    def _make_config_list(self):
        self.config_list = []
        if self.random_temperature:
            temperature_list = np.random.uniform(np.clip(self.temperature-0.5, 0.0, 2.0), np.clip(self.temperature+0.5, 0.0, 2.0), self.n_agent)
        else:
            temperature_list = [self.temperature] * self.n_agent

        if self.random_gripper:
            gripper_list = [random.choice([False]*1 + [True]*1) for _ in range(self.n_agent)]
        else:
            gripper_list = [self.enable_gripper] * self.n_agent

        for i in range(self.n_agent):
            # Create a new config for each agent
            config = {
                'enable_gripper': gripper_list[i],
                'gemini_config': {
                    'response_mime_type': 'application/json',
                    'response_schema': InitialResponseSchema,
                    'temperature': temperature_list[i],
                }
            }
            self.config_list.append(config)

    def _save_sampling_results(self, results):
        # Create a directory for the current timestamp
        base_save_folder = os.path.join(self.base_path, "logs", self.timestamp)
        sample_save_folder = os.path.join(base_save_folder, "samples")
        os.makedirs(sample_save_folder, exist_ok=True)
        for i, res in enumerate(results):
            try:
                run_file = os.path.join(sample_save_folder, f"sample_id_{i}.json")
                with open(run_file, 'w') as f:
                    json.dump(res, f, indent=4)
            except Exception as e:
                print(f'Error in saving samples: {e}. \nSkipping sample from agent {i}')
                continue
        print(f"Results saved in {base_save_folder}")
        return base_save_folder
    
    def _processing_sampling_results(self, results):
        strategies = []
        for i in range(len(results)):
            try:
                # Save the results
                single_agent_results = results[i]
                json_content = json.loads(single_agent_results)
                strats = json_content['strategies']
                corresponding_config = self.config_list[i]
                del corresponding_config['gemini_config']['response_schema']
                strats = [
                    {**strat, **corresponding_config}
                    for strat in strats
                ]
                strategies += strats
            except Exception as e:
                print(f'Error in processing samples: {e}. \nSkipping sample from agent {i}')
                continue
        print(f"Total {len(strategies)} tool strategies received.")
        return strategies

    def begin_design_and_action_sampling(self):
        while True:
            try: 
                # Handling all prompts and attachments
                self.sc = SamplingCollection(self.n_agent, self.max_n_query_threads, self.model_name)
                prompts = self._get_prompts()
                agent_configs = [single_config['gemini_config'] for single_config in self.config_list]
                # The prompts is a list of prompt
                results = self.sc.start_sampling_agent(prompts, agent_configs)
                break
            except Exception as e:
                print(f'Error in design and action sampling: {e}. \nRestarting')
                continue
        strategies = self._processing_sampling_results(results)
        base_save_folder = self._save_sampling_results(strategies)
        return base_save_folder

    def evolve_design_and_action(self, previous_designs: list[str]):
        while True:
            try: 
                # Handling all prompts and attachments
                self.sc = SamplingCollection(self.n_agent, self.max_n_query_threads, self.model_name)
                prompts = self._get_prompts(previous_designs=previous_designs)
                agent_configs = [single_config['gemini_config'] for single_config in self.config_list]
                # The prompts is a list of prompt
                results = self.sc.start_sampling_agent(prompts, agent_configs)
                break
            except Exception as e:
                print(f'Error in design and action sampling: {e}. \nRestarting')
                continue
        strategies = self._processing_sampling_results(results)
        base_save_folder = self._save_sampling_results(strategies)
        return base_save_folder

    def _get_prompts(self, previous_designs = None):
        if previous_designs:
            prompts = self.prompt_composer.create_evolve_design_and_action_prompt_list(
                n_agents=self.n_agent,
                config_list=self.config_list,
                previous_designs=previous_designs
            )
        else:
            prompts = self.prompt_composer.create_design_and_action_prompt_list(
                n_agents=self.n_agent,
                config_list=self.config_list,
            )
        return prompts
    
    