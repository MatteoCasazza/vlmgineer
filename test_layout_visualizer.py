import json
import glob
import os
import pybullet as p
import pybullet_data


def find_latest_layout_results():
    files = glob.glob("layout_results_*.json")

    if not files:
        raise FileNotFoundError("No layout_results_*.json file found in the current folder.")

    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def get_color(object_type):
    colors = {
        "robot_base": [0.2, 0.2, 0.2, 1.0],
        "table": [0.55, 0.35, 0.2, 1.0],
        "task_object": [1.0, 0.0, 0.0, 1.0],
        "target_area": [0.0, 0.8, 0.0, 0.6],
        "obstacle": [0.0, 0.0, 1.0, 1.0],
    }

    return colors.get(object_type, [0.8, 0.8, 0.8, 1.0])


def spawn_box(obj):
    name = obj["object_name"]
    object_type = obj["object_type"]
    position = obj["position"]
    orientation_rpy = obj["orientation"]
    dimensions = obj["dimensions"]

    half_extents = [
        dimensions[0] / 2,
        dimensions[1] / 2,
        dimensions[2] / 2,
    ]

    orientation_quat = p.getQuaternionFromEuler(orientation_rpy)
    color = get_color(object_type)

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation_quat,
    )

    p.addUserDebugText(
        text=name,
        textPosition=[position[0], position[1], position[2] + dimensions[2] / 2 + 0.05],
        textSize=1.0,
        lifeTime=0,
    )

    return body_id


def draw_world_axes():
    axis_length = 1.0

    p.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [1, 0, 0], lineWidth=3)
    p.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 1, 0], lineWidth=3)
    p.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], lineWidth=3)

    p.addUserDebugText("X", [axis_length, 0, 0], textSize=1.2)
    p.addUserDebugText("Y", [0, axis_length, 0], textSize=1.2)
    p.addUserDebugText("Z", [0, 0, axis_length], textSize=1.2)


def snap_objects_to_table_surface(layout):
    tables = [obj for obj in layout["objects"] if obj["object_type"] == "table"]

    if not tables:
        return layout

    table = tables[0]
    table_top_z = table["position"][2] + table["dimensions"][2] / 2.0

    surface_object_types = ["task_object", "target_area", "obstacle"]

    for obj in layout["objects"]:
        if obj["object_type"] in surface_object_types:
            obj_height = obj["dimensions"][2]
            obj["position"][2] = table_top_z + obj_height / 2.0

    return layout

def main():
    results_file = find_latest_layout_results()

    print(f"Loading layout results from: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    best_layout = data["best_layout"]["layout"]
    best_layout = snap_objects_to_table_surface(best_layout)

    print("\nVisualizing best layout:")
    print(best_layout["layout_name"])
    print(best_layout["layout_description"])

    physics_client = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    draw_world_axes()

    robot_id = None
    object_body_ids = {}
    object_types = {}

    for obj in best_layout["objects"]:
        if obj["object_type"] == "robot_base":
            robot_base_position = [
                obj["position"][0],
                obj["position"][1],
                0.0,
            ]

            print("Loading Panda robot at:", robot_base_position)

            robot_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=robot_base_position,
                baseOrientation=p.getQuaternionFromEuler(obj["orientation"]),
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            )
        else:
            body_id = spawn_box(obj)
            object_body_ids[obj["object_name"]] = body_id
            object_types[obj["object_name"]] = obj["object_type"]

    if robot_id is not None:
        print("\nVisualizer mode: scene loaded.")
        print("Physical validation is handled by PyBulletLayoutEvaluator.")
    else:
        print("No robot found.")

    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=45,
        cameraPitch=-35,
        cameraTargetPosition=[0.5, 0.0, 0.4],
    )

    print("\nPyBullet window opened.")
    print("Press ENTER in this terminal to close the visualization.")
    input()

    p.disconnect(physics_client)


if __name__ == "__main__":
    main()