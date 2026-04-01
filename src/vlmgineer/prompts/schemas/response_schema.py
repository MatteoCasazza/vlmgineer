from pydantic import BaseModel
from numpydantic import NDArray, Shape

class _StrategySchema(BaseModel):
    strategy_name: str
    strategy_description: str
    tool_id: int
    tool_name: str
    tool_urdf: str
    tool_description: str
    action_sets: list[NDArray[Shape["* num_waypoints, * waypoint"], float]]
    action_description: str

class InitialResponseSchema(BaseModel):
    scene_analysis: str
    task_analysis: str
    strategies: list[_StrategySchema]

class HumanResponseSchema(BaseModel):
    enable_gripper: bool
    action_sets: list[NDArray[Shape["* num_waypoints, * waypoint"], float]]
    action_description: str
    strategy_description: str
    tool_id: int
    tool_name: str
    tool_urdf: str
    tool_description: str

class NoToolResponseSchema(BaseModel):
    gripper: bool
    orientation: bool
    action_sets: list[NDArray[Shape["* num_waypoints, * waypoint"], float]]
    action_description: str
    strategy_description: str

class ActionResponseSchema(BaseModel):
    action_sets: list[NDArray[Shape["* num_waypoints, * waypoint"], float]]
    action_description: str
