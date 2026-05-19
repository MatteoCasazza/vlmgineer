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


def check_robot_reachability(robot_id, objects, object_body_ids, object_types):
    ee_link_index = 11  # Franka Panda end-effector link in pybullet_data model

    reachable_results = []

    for obj in objects:
        if obj["object_type"] not in ["task_object", "target_area"]:
            continue

        target_pos = obj["position"]

        ik_solution = p.calculateInverseKinematics(
            robot_id,
            ee_link_index,
            target_pos,
        )

        if ik_solution is None or len(ik_solution) == 0:
            reachable = False
            error = None

            forbidden_collisions = []
            allowed_contacts = []
            
            collision_free = False

        else:
            for joint_idx in range(7):
                p.resetJointState(robot_id, joint_idx, ik_solution[joint_idx])

            p.stepSimulation()

            ee_state = p.getLinkState(robot_id, ee_link_index)
            ee_pos = ee_state[0]

            error = (
                (ee_pos[0] - target_pos[0]) ** 2
                + (ee_pos[1] - target_pos[1]) ** 2
                + (ee_pos[2] - target_pos[2]) ** 2
            ) ** 0.5

            reachable = error < 0.08

            forbidden_collisions, allowed_contacts = check_robot_collisions(
                robot_id,
                object_body_ids,
                object_types,
            )

            collision_free = len(forbidden_collisions) == 0

        physically_feasible = reachable and collision_free

        reachable_results.append({
            "object_name": obj["object_name"],
            "object_type": obj["object_type"],
            "target_position": target_pos,
            "reachable": reachable,
            "ik_error": error,
            "collision_free": collision_free,
            "forbidden_collisions": forbidden_collisions,
            "allowed_contacts":allowed_contacts,
            "physically_feasible": physically_feasible,
        })

    return reachable_results


def check_robot_collisions(robot_id, object_body_ids, object_types):
    p.performCollisionDetection()

    forbidden_collisions = []
    allowed_contacts = []

    allowed_contact_types = ["task_object", "target_area"]

    for obj_name, body_id in object_body_ids.items():
        contact_points = p.getContactPoints(robot_id, body_id)

        if len(contact_points) == 0:
            continue

        obj_type = object_types[obj_name]

        contact_info = {
            "object_name": obj_name,
            "object_type": obj_type,
            "num_contacts": len(contact_points),
        }

        if obj_type in allowed_contact_types:
            allowed_contacts.append(contact_info)
        else:
            forbidden_collisions.append(contact_info)

    return forbidden_collisions, allowed_contacts


def main():
    results_file = find_latest_layout_results()

    print(f"Loading layout results from: {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    best_layout = data["best_layout"]["layout"]

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
            print("Loading Panda robot at:", obj["position"])
            robot_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=obj["position"],
                baseOrientation=p.getQuaternionFromEuler(obj["orientation"]),
                useFixedBase=True,
            )
        else:
            body_id = spawn_box(obj)
            object_body_ids[obj["object_name"]] = body_id
            object_types[obj["object_name"]] = obj["object_type"]

    if robot_id is not None:
        reachability_results = check_robot_reachability(
            robot_id,
            best_layout["objects"],
            object_body_ids,
            object_types,
        )

        print("\n=== TRUE IK + COLLISION FEASIBILITY CHECK ===")
        for result in reachability_results:
            error_text = "None" if result["ik_error"] is None else f"{result['ik_error']:.4f}"

            print(
                f"{result['object_name']} | "
                f"reachable={result['reachable']} | "
                f"ik_error={error_text} | "
                f"collision_free={result['collision_free']} | "
                f"physically_feasible={result['physically_feasible']}"
            )

            if result["allowed_contacts"]:
                print("  Allowed contacts:")
                for col in result["allowed_contacts"]:
                    print(
                        f"    - {col['object_name']} "
                        f"[{col['object_type']}] "
                        f"({col['num_contacts']} contacts)"
                    )

            if result["forbidden_collisions"]:
                print("  Forbidden collisions:")
                for col in result["forbidden_collisions"]:
                    print(
                        f"    - {col['object_name']} "
                        f"[{col['object_type']}] "
                        f"({col['num_contacts']} contacts)"
                    )
    else:
        print("No robot found, skipping IK reachability check.")

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