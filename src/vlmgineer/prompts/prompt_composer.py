import numpy as np
import os
import mimetypes
import io
import json
from pathlib import Path
from google.genai import types
import time
from dataclasses import dataclass
import vlmgineer.prompts.prompt_utils as prompt_utils

@dataclass
class SinglePrompt:
    instruction_prompts : str
    file_prompts : list[str]

class PromptComposer:
    def __init__(self, **kwargs):
        self.task_name = kwargs.get('task_name')
        self.n_tool_samples_batch_size = kwargs.get('n_tool_samples_batch_size', 5)
        self.n_action_samples_batch_size = kwargs.get('n_action_samples_batch_size', 5)
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def get_basic_intro_prompt(self):
        INTRO_PROMPT = """You are a robotics hardware and controls expert. You operate with boldness and brilliance in the physical realm. You work with a robot arm that sits in the origin of your environment. You will be presented with some robotic task, and will be asked to design tools and actions to complete the task. Your goal is not to complete the task to perfection in one fell swoop. Instead, your meta-goal is to generate a wide range of differentiated good solutions over time where one of them will inevitably succeed."""
        return INTRO_PROMPT
    
    def get_evolution_intro_prompt(self):
        EVOLUTION_INTRO_PROMPT = """You are a robotics hardware and controls expert. You operate with boldness and brilliance in the physical realm. The goal is to create tools and actions to complete a given task. You will be given a list of previously generated tool designs via JSON with URDF. Your goal is to evolve the tool designs via mutation and crossover, and generate the new best actions for the evolved tools. This will be done in a way that is similar to genetic algorithms, and will be specified in detail in the "Evolutionary Process" section below."""
        return EVOLUTION_INTRO_PROMPT

    def get_tool_specification_prompt(self, config: dict):
        enable_gripper = config.get('enable_gripper', False)
        TOOL_GRIPPER_SPECIFICATION_PROMPT = """(Tool Specifications) Your design of the tool must follow these rules: (1) You must only use 3D rectangles for each component; (2) Your tool will be outputted in a URDF block format, which should be directly added to the end of a panda URDF file, before the robot closing declaration; (3) Make sure your tools weigh very little in the URDF file, where each tool part should weigh no more than a few grams (these weights do not have to be realistic, it is just for the robot inverse kinematics to have a easier time converging).  (4) Your design will be a pair of attachment to the robot gripper fingers (which allows the tool to be acuated with the robot gripper); You should attached the left attachment to "panda_leftfinger" and the right attachment to "panda_rightfinger". (5) Any attachments you design should geometrically be directly connected to its parent links in the URDF (there should be no gaps in between!) (6) As a general observation, you perform better when the tools you design are complex and intricate."""
        TOOL_RIGID_SPECIFICATION_PROMPT = """(Tool Specifications) Your design of the tool must follow these rules: (1) You must only use 3D rectangles for each component; (2) Your tool will be outputted in a URDF block format, which should be directly added to the end of a panda URDF file, before the robot closing declaration; (3) Make sure your tools weigh very little in the URDF file, where each tool part should weigh no more than a few grams (these weights do not have to be realistic, it is just for the robot inverse kinematics to have a easier time converging). (4) Your design will be a single rigid tool, which should be attached directly to the "panda_virtual" link, which you can safely assume to have the same orientation as the world frame. (5) Any attachments you design should geometrically be directly connected to its parent links in the URDF (there should be no gaps in between!) (6) As a general observation, you perform better when the tools you design are complex and intricate."""
        return TOOL_GRIPPER_SPECIFICATION_PROMPT if enable_gripper else TOOL_RIGID_SPECIFICATION_PROMPT
    
    def get_procedure_prompt(self):
        PROCEDURE_PROMPT = f"""The procedure you will follow:
        1. Receive Environment Descriptions: The user will provide some detailed environment description, robotic task instruction, and an initial image of the workspace area from the overhead camera.
        2. Describe the Scene: Analyze the environment. Write down the spatial relationship, including by not limited to the position, orientation, dimension, and geometry of all the objects in the scene. Use all information provided to you, including all text, code, and image.
        3. Create Strategies and Designs: You will need to create {self.n_tool_samples_batch_size} tool that you can use to complete the task. For each of the tools you designed, you must generate {self.n_action_samples_batch_size} set of action waypoints that you can use to complete the task. Specifically, for a total of {self.n_tool_samples_batch_size} times, do the following steps:
            (a) First, write down a completely different, out-of-the-box tool designs to tackle the task. Make it unlike any other tool design you made in your other strategies.
            (b) Create this tools following the "Tool Specification" section below.
            (c) For this tool, write the following down: (1) The spatial relationship (pose transformation) between the end-effector and each component of the tool; (2) The 3D space that each tool component will take up when connected to the robot; (3) The usage of each component of the tool when carrying out the task.
            (d) Use your previous analysis to tweak any obvious issues with the position, orientation, and dimension of your tool design.
            (e) Next, using your knowledge of the tool and your in depth analysis regarding the intricate 3D spatial relationships between the tool and its environment, create {self.n_action_samples_batch_size} number of different step by step action plans to enable to effective tool use (See more in "Desired Action Criteria Definitions"). Be very wary about how objects interacts with each other!
            (f) Transform your step-by-step action plan into waypoints adhering to the "Action Specifications". During this transformation, think about the inherent nature of controlling robots with waypoint control and the difficulty that may present."""
        return PROCEDURE_PROMPT

    def get_desired_action_criteria_definitions_prompt(self):
        DESIRED_ACTION_CRITERIA_DEFINITIONS_PROMPT = f"""(Desired Action Criteria Definitions) For the description below, we will call a single sequential set of waypoints in a single rollout as one "action set". For each tool you created, the goal is to generate {self.n_action_samples_batch_size} action sets that optimizes the task-successfulness and motion differentiation. Task-successfulness is optimized when an action set is able to complete the task successfully. Motion differentiation is optimized when there exists a large variance in the motion taken across all action sets you design for the same tool. A large variance in motion is defined the tool, at each time step, is located at a different location in the 3D space.Think about how a tool can be used to interact with the object from many different sides, angles, and ways. When both conditions are met, you have successfully designed a good set of action sets."""
        return DESIRED_ACTION_CRITERIA_DEFINITIONS_PROMPT
    
    def get_action_specification_prompt(self, config: dict):
        enable_gripper = config.get('enable_gripper', False)
        ACTION_GRIPPER_SPECIFICATION_PROMPT = """(Action Specifications) Your tool-using action will be a Nx7 numpy array of action waypoints, where N is the number of waypoints, and each waypoint is of dimension 7 (xyz position + roll-pitch-yaw euler angle orientations + binary gripper open/close state in integers [0 for open, 1 for close]). Your action needs to be precisely seven numbers per waypoint. Your waypoints will be carried out by the EnvRunner class. It is important to stress this: the action waypoints are controlling the robot end-effector "panda_virtual" link: this means you have to carefully take into account the dimensions of the tool and the thickness of its parts when designing effective waypoints. Again, you can safely assume the end-effector has the same orientation as the world frame upon initialization (see frame clarification again for details)!"""
        ACTION_RIGID_SPECIFICATION_PROMPT = """(Action Specifications) Your tool-using action will be a Nx6 numpy array of action waypoints, where N is the number of waypoints, and each waypoint is of dimension 6 (xyz position + roll-pitch-yaw euler angle orientations). Your action needs to be precisely six numbers per waypoint. Your waypoints will be carried out by the EnvRunner class. It is important to stress this: the action waypoints are controlling the robot end-effector "panda_virtual" link: this means you have to carefully take into account the dimensions of the tool and the thickness of its parts when designing effective waypoints. Again, you can safely assume the end-effector has the same orientation as the world frame upon initialization (see frame clarification again for details)!"""
        return ACTION_GRIPPER_SPECIFICATION_PROMPT if enable_gripper else ACTION_RIGID_SPECIFICATION_PROMPT
    
    def get_robot_workspace_specification_prompt(self):
        ROBOT_WORKSPACE_SPECIFICATION_PROMPT = """(Robot Workspace Specification) The robot workspace is a semi-circle with the following dimensions: 
        - The center of the semi-circle is at the origin of the robot frame.
        - The radius of the semi-circle is 0.8 meters.
        - It points towards the positive z direction, meaning it spans from the positive to negative x and y axis, and only toward the positive z direction.
        This workspace constrain should make you think about which regions in the environment are accessible to the robot, and which are not. Your tool should be there to help the robot interact with those regions that are not accessible!"""
        return ROBOT_WORKSPACE_SPECIFICATION_PROMPT

    def get_tool_collision_specification_prompt(self):
        TOOL_COLLISION_PROMPT = """(Tool Collision On-Spawn) When the tool spawns as an attachment to the robot end-effector, it must adhere by the following collision constraints:
        (1) It must NOT spawn into parts of the robot: The robot always begins with its end-effector pointing downwards. This mean the robot body parts around the end-effector are concentrated above the end-effector. Your tool must not collide with the robot body parts, and make sure bypass the robot body parts (e.g. creating C-shape bends around the robot instead of a straight line, or creating zig-zags around the robot instead of a straight line, etc.).
        (2) It must NOT spawn into parts of the environment: Any objects in the environment, if present,including but not limited to the table, the floor, and any objects on top of the table or the floor, are not allowed to be touching the tool that you design. Make smart choices about the placement and dimension of your tool to avoid these collisions.
        """
        return TOOL_COLLISION_PROMPT

    def get_frame_clarification_prompt(self):
        FRAME_CLARIFICATION_PROMPT = """(Frame Clarification) In the world frame, front/back is along the x axis, left/right is along the y axis and up/down is along the z axis with following directions: Positive x: Towards the front of the table. Negative x: Towards the back of the table. Positive y: Towards the left. Negative y: Towards the right. Positive z: Up, towards the ceiling. Negative z: Down, towards the floor. In terms of orientation, starting from the origin frame: Positive rotation about the x-axis: tilting end-effector head to the left. Negative rotation about the x-axis: tilting end-effector head to the right. Positive rotation about the y-axis: tilting end-effector head down. Negative rotation about the y-axis: tilting end-effector head up. Positive rotation about the z-axis: rotating the end-effector head counter-clockwise. Negative rotation about the z-axis: rotating the end-effector head clockwise."""
        return FRAME_CLARIFICATION_PROMPT
    
    def get_output_format_prompt(self):
        OUTPUT_FORMAT_PROMPT = """(Output Format) For each strategy, you will output the following:
        - strategy_name: name for the strategy
        - strategy_description: a short description of the strategy you intend for these tool and actions.
        - tool_id: the id of the tool
        - tool_name: the name of the tool"""
        return OUTPUT_FORMAT_PROMPT

    def get_unit_prompt(self):
        UNIT_PROMPT = """(Units) All numbers described is in meters."""
        return UNIT_PROMPT

    def get_optional_prompt(self):
        OPTIONAL_PROMPT = """(Optional) If you wish, you can make the tools pretty by giving it different colors and transparency in the urdf output."""
        return OPTIONAL_PROMPT

    def get_evolutionary_process_prompt(self):
        EVOLUTIONARY_PROCESS_PROMPT = f"""(Evolutionary Process) Your design decision is a part of a tool design genetic algorithm. For each of the {self.n_tool_samples_batch_size} tool designs, you can choose to either mutate or crossover. Specifically, tool mutation is defined as one change to a single randomly selected previous tool design. Mutation changes include:
        (1) Changing the dimension, location, or orientation of a single component of the tool.
        (2) Adding, removing, or replacing a single component of the tool.
        Crossover is defined as the process of combining two randomly selected previous tool designs to create a new tool design. Combination is defined as:
        (1) Selecting components from two previous tool designs and combining them to form a new tool design. 
        All mutation and crossover decisions must potentially increase the likelihood of task-successfulness, yet all decisions must be different and diverse."""
        return EVOLUTIONARY_PROCESS_PROMPT
    
    def create_single_design_and_action_prompt(self, config: dict):
        # Prepare universal attachments
        file_prompts = []
        file_prompts += prompt_utils.get_universal_attachment_paths(self.base_path, config)
        file_prompts += prompt_utils.get_task_attachment_paths(self.base_path, self.task_name)
        instruction_prompts = self.get_initial_instruction_prompt(config)
        return SinglePrompt(instruction_prompts=instruction_prompts, file_prompts=file_prompts)
    
    def create_evolve_design_and_action_prompt(self, config: dict, previous_designs: list[str]):
        # Prepare universal attachments
        file_prompts = []
        file_prompts += prompt_utils.get_universal_attachment_paths(self.base_path, config)
        file_prompts += prompt_utils.get_task_attachment_paths(self.base_path, self.task_name)
        instruction_prompts = self.get_evolution_instruction_prompt(config, previous_designs)
        return SinglePrompt(instruction_prompts=instruction_prompts, file_prompts=file_prompts)

    def create_design_and_action_prompt_list(self, n_agents, config_list: list[dict] = None):
        if config_list:
            assert len(config_list) == n_agents, "Number of configs must be equal to number of agents"
        else:
            config_list = [{}] * n_agents
        agent_prompts = []
        for i in range(n_agents):
            single_prompt = self.create_single_design_and_action_prompt(config_list[i])
            agent_prompts.append(single_prompt)
        return agent_prompts

    def create_evolve_design_and_action_prompt_list(self, n_agents, config_list: list[dict] = None, previous_designs: list[str] = None):
        if config_list:
            assert len(config_list) == n_agents, "Number of configs must be equal to number of agents"
        else:
            config_list = [{}] * n_agents
        agent_prompts = []
        for i in range(n_agents):
            single_prompt = self.create_evolve_design_and_action_prompt(config_list[i], previous_designs)
            agent_prompts.append(single_prompt)
        return agent_prompts

    def get_initial_instruction_prompt(self, config: dict):
        return f"""
        {self.get_basic_intro_prompt()}

        {self.get_procedure_prompt()}

        {self.get_tool_specification_prompt(config)}

        {self.get_desired_action_criteria_definitions_prompt()}

        {self.get_action_specification_prompt(config)}

        {self.get_frame_clarification_prompt()}

        {self.get_robot_workspace_specification_prompt()}

        {self.get_tool_collision_specification_prompt()}

        {self.get_output_format_prompt()}

        {self.get_unit_prompt()}

        {self.get_optional_prompt()}
        """

    def get_evolution_instruction_prompt(self, config: dict, previous_designs: list[str]):
        return f"""
        {self.get_evolution_intro_prompt()}

        Here are the previous tool designs:
        {previous_designs}

        {self.get_procedure_prompt()}

        {self.get_evolutionary_process_prompt()}

        {self.get_tool_specification_prompt(config)}

        {self.get_desired_action_criteria_definitions_prompt()}

        {self.get_action_specification_prompt(config)}

        {self.get_frame_clarification_prompt()}

        {self.get_robot_workspace_specification_prompt()}

        {self.get_tool_collision_specification_prompt()}

        {self.get_output_format_prompt()}

        {self.get_unit_prompt()}

        {self.get_optional_prompt()}
        """