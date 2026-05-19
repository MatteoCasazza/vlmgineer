from dataclasses import dataclass


@dataclass
class LayoutPrompt:
    instruction_prompts: str
    file_prompts: list[str]


class LayoutPromptComposer:
    def __init__(self, n_layout_samples: int = 3):
        self.n_layout_samples = n_layout_samples

    def get_intro_prompt(self) -> str:
        return """
You are a robotics work-cell design expert.
Your task is to propose physically meaningful robotic work-cell layouts.
You must reason about object placement, robot reachability, collisions, workspace usage, and task-driven spatial constraints.
You do not design tools or robot actions.
You only design candidate layouts.
"""

    def get_layout_procedure_prompt(self) -> str:
        return f"""
The procedure you will follow:

1. Analyze the task and scene.
2. Identify all relevant objects, workstations, targets, obstacles, and robot-related constraints.
3. Generate {self.n_layout_samples} diverse candidate work-cell layouts.
4. For each layout, specify:
   - object names
   - object types
   - 3D positions in meters
   - orientations in roll-pitch-yaw radians
   - bounding-box dimensions in meters
   - functional role of each object
   - spatial relations between objects
   - constraints considered
   - expected advantages
5. Prefer layouts that are reachable, compact, collision-free, and easy to validate in simulation.
"""

    def get_workspace_prompt(self) -> str:
        return """
Workspace assumptions:
- The robot base is initially located at the world origin [0, 0, 0].
- All units are meters.
- The work-cell is represented in a right-handed 3D coordinate frame.
- Objects should remain inside a reasonable workspace around the robot.
- Avoid object overlap.
- Keep task-relevant objects inside the robot reachable workspace when possible.

Important physical constraints:
- The table must not intersect or surround the robot base.
- The table center should usually be at least 0.8 m in front of the robot base.
- The closest edge of the table must keep at least 0.25 m clearance from the robot base footprint.
- Pick and place objects should lie on the table surface, but the robot base must remain outside the table footprint.
- Prefer layouts where the robot can approach objects from above without the arm colliding with the table.
- Task objects and target areas should be placed within 0.45 m to 0.75 m from the robot base in the XY plane.
- Avoid placing pick/place locations farther than 0.8 m from the robot base.
- Prefer table positions that allow the nearest task objects to remain reachable while preserving robot-table clearance.
- Pick and place locations should be at least 0.15 m away from table edges.
- Avoid placing pick/place targets too close to table borders or obstacles.
- The robot should be able to approach each task object from above with at least 0.05 m clearance from forbidden objects.
- Prefer pick/place positions near the front half of the table, not deep inside the tabletop.

Additional robot feasibility constraints:
- Pick and place objects should be located in front of the robot, not too close to the robot base.
- Prefer object XY distances from the robot base between 0.35 m and 0.60 m.
- Avoid placing objects directly under the robot wrist or too close to the robot base.
- Leave enough free space above each task object for a top-down end-effector approach.
- For Panda-like robot validation, prefer targets near the center-front region of the table.
- Avoid very compact corner layouts where the robot wrist must fold into the table.
- Keep at least 0.10 m free horizontal clearance around each pick/place object.
"""

    def get_output_format_prompt(self) -> str:
        return """
Output format:
Return a JSON object matching the LayoutResponseSchema.
Do not include free text outside the JSON.
"""

    def create_layout_prompt(self) -> LayoutPrompt:
        instruction = f"""
{self.get_intro_prompt()}

{self.get_layout_procedure_prompt()}

{self.get_workspace_prompt()}

{self.get_output_format_prompt()}
"""
        return LayoutPrompt(
            instruction_prompts=instruction,
            file_prompts=[],
        )