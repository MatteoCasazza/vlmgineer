import json
import glob
import os
import math

import pybullet as p
import pybullet_data


EE_LINK_INDEX = 11


def find_latest_layout_results():
    files = glob.glob("layout_results_*.json")

    if not files:
        raise FileNotFoundError("No layout_results_*.json file found.")

    return max(files, key=os.path.getmtime)


def get_color(object_type):
    colors = {
        "robot_base": [0.2, 0.2, 0.2, 1.0],
        "table": [0.55, 0.35, 0.2, 1.0],
        "task_object": [1.0, 0.0, 0.0, 1.0],
        "target_area": [0.0, 0.8, 0.0, 0.6],
        "obstacle": [0.0, 0.0, 1.0, 1.0],
    }

    return colors.get(object_type, [0.8, 0.8, 0.8, 1.0])


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

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=get_color(object_type),
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=p.getQuaternionFromEuler(orientation_rpy),
    )

    p.addUserDebugText(
        text=name,
        textPosition=[
            position[0],
            position[1],
            position[2] + dimensions[2] / 2 + 0.05,
        ],
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


def reset_robot_home(robot_id):
    home_joints = [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8]

    for joint_idx, joint_value in enumerate(home_joints):
        p.resetJointState(robot_id, joint_idx, joint_value)

    p.stepSimulation()


def get_target_pose_above_object(obj):
    object_top_z = obj["position"][2] + obj["dimensions"][2] / 2.0

    target_pos = [
        obj["position"][0],
        obj["position"][1],
        object_top_z + 0.18,
    ]

    target_orientation = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

    return target_pos, target_orientation


def compute_ik_pose(robot_id, obj):
    target_pos, target_orientation = get_target_pose_above_object(obj)

    reset_robot_home(robot_id)

    ik_solution = p.calculateInverseKinematics(
        robot_id,
        EE_LINK_INDEX,
        target_pos,
        targetOrientation=target_orientation,
        maxNumIterations=100,
        residualThreshold=1e-4,
    )

    if ik_solution is None or len(ik_solution) == 0:
        return None, target_pos

    return list(ik_solution[:7]), target_pos


def check_forbidden_contacts(robot_id, object_body_ids, object_types):
    p.performCollisionDetection()

    forbidden_types = ["table", "obstacle"]
    contacts = []

    for obj_name, body_id in object_body_ids.items():
        obj_type = object_types[obj_name]

        if obj_type not in forbidden_types:
            continue

        contact_points = p.getContactPoints(robot_id, body_id)

        if contact_points:
            contacts.append({
                "object_name": obj_name,
                "object_type": obj_type,
                "num_contacts": len(contact_points),
            })

    return contacts


def main():
    results_file = find_latest_layout_results()
    print(f"Loading layout results from: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    best_layout = data["best_layout"]["layout"]
    best_layout = snap_objects_to_table_surface(best_layout)

    print("\nVisualizing IK poses for best layout:")
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

    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=45,
        cameraPitch=-35,
        cameraTargetPosition=[0.5, 0.0, 0.4],
    )

    if robot_id is None:
        print("No robot found.")
        input("Press ENTER to close.")
        p.disconnect(physics_client)
        return

    target_objects = [
        obj for obj in best_layout["objects"]
        if obj["object_type"] in ["task_object", "target_area"]
    ]

    print("\nScene loaded.")
    print("Press ENTER to show IK poses one by one.")
    input()

    for obj in target_objects:
        print(f"\nComputing IK for: {obj['object_name']}")

        ik_joints, target_pos = compute_ik_pose(robot_id, obj)

        if ik_joints is None:
            print(f"No IK solution found for {obj['object_name']}")
            continue

        for joint_idx, joint_value in enumerate(ik_joints):
            p.resetJointState(robot_id, joint_idx, joint_value)

        p.stepSimulation()

        p.addUserDebugText(
            text=f"IK pose: {obj['object_name']}",
            textPosition=[
                target_pos[0],
                target_pos[1],
                target_pos[2] + 0.1,
            ],
            textSize=1.2,
            lifeTime=0,
        )

        p.addUserDebugLine(
            [target_pos[0], target_pos[1], target_pos[2]],
            [target_pos[0], target_pos[1], target_pos[2] + 0.15],
            [1, 0, 1],
            lineWidth=3,
        )

        contacts = check_forbidden_contacts(
            robot_id,
            object_body_ids,
            object_types,
        )

        if contacts:
            print("Forbidden contacts detected:")
            for contact in contacts:
                print(
                    f"  - {contact['object_name']} "
                    f"[{contact['object_type']}] "
                    f"({contact['num_contacts']} contacts)"
                )
        else:
            print("No forbidden contacts detected in this IK pose.")

        input("Press ENTER for next IK pose...")

    print("\nFinished IK pose visualization.")
    print("Press ENTER to close.")
    input()

    p.disconnect(physics_client)


if __name__ == "__main__":
    main()