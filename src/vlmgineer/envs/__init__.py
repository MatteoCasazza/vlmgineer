from vlmgineer.envs.bring_cube_closer.bring_cube_closer_env import BringCubeCloserEnv
from vlmgineer.envs.elevate_plate.elevate_plate_env import ElevatePlateEnv
from vlmgineer.envs.move_ball.move_ball_env import MoveBallEnv
from vlmgineer.envs.collect_and_elevate_spheres.collect_and_elevate_spheres_env import CollectAndElevateSpheresEnv
from vlmgineer.envs.dislodge_cube.dislodge_cube_env import DislodgeCubeEnv
from vlmgineer.envs.place_cubes_inside.place_cubes_inside_env import PlaceCubesInsideEnv
from vlmgineer.envs.gather_strawberries.gather_strawberries_env import GatherStrawberriesEnv
from vlmgineer.envs.clean_table_top.clean_table_top_env import CleanTableTopEnv
from vlmgineer.envs.close_plier.close_plier_env import ClosePlierEnv
from vlmgineer.envs.bring_screen_down.bring_screen_down_env import BringScreenDownEnv
from vlmgineer.envs.puck_to_goal.puck_to_goal_env import PuckToGoalEnv
from vlmgineer.envs.take_one_book_out.take_one_book_out_env import TakeOneBookOutEnv
from vlmgineer.envs.retrieve_high_object.retrieve_high_object_env import RetrieveHighObjectEnv
from vlmgineer.envs.rlbench_hockey.hockey_env import HockeyEnv
from vlmgineer.envs.rlbench_scoop_with_spatula.scoop_with_spatula_env import ScoopWithSpatulaEnv
from vlmgineer.envs.rlbench_sweep_to_dustpan.sweep_to_dustpan_env import SweepToDustpanEnv
from vlmgineer.envs.rlbench_reach_and_drag.reach_and_drag_env import ReachAndDragEnv
from vlmgineer.envs.serve_turkey_legs.serve_turkey_legs_env import ServeTurkeyLegsEnv
from vlmgineer.envs.take_cookie_out_from_jar.take_cookie_out_from_jar_env import TakeCookieOutFromJarEnv
from vlmgineer.envs.lift_box.lift_box_env import LiftBoxEnv
from vlmgineer.envs.lift_chips.lift_chips_env import LiftChipsEnv

env_dict = {
    "bring_cube_closer": BringCubeCloserEnv,
    "elevate_plate": ElevatePlateEnv,
    "move_ball": MoveBallEnv,
    "collect_and_elevate_spheres": CollectAndElevateSpheresEnv,
    "dislodge_cube": DislodgeCubeEnv,
    "place_cubes_inside": PlaceCubesInsideEnv,
    "gather_strawberries": GatherStrawberriesEnv,
    "clean_table_top": CleanTableTopEnv,
    "close_plier": ClosePlierEnv,
    "bring_screen_down": BringScreenDownEnv,
    "puck_to_goal": PuckToGoalEnv,
    "take_one_book_out": TakeOneBookOutEnv,
    "retrieve_high_object": RetrieveHighObjectEnv,
    "rlbench_hockey": HockeyEnv,
    "rlbench_scoop_with_spatula": ScoopWithSpatulaEnv,
    "rlbench_sweep_to_dustpan": SweepToDustpanEnv,
    "rlbench_reach_and_drag": ReachAndDragEnv,
    "serve_turkey_legs": ServeTurkeyLegsEnv,
    "take_cookie_out_from_jar": TakeCookieOutFromJarEnv,
    "lift_box": LiftBoxEnv,
    "lift_chips": LiftChipsEnv,
}

def get_env_class(task_name: str):
    return env_dict[task_name]