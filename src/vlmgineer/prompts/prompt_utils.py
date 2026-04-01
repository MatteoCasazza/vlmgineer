import os
import mimetypes
import io
import json
from pathlib import Path
from google.genai import types
import time
import cv2


def get_base_env_path(base_path: str) -> str:
    """
    Return the path to the base_env.py file.
    """
    root = os.path.join(base_path, "envs", "base_env.py")
    if os.path.isfile(root):
        return root
    else:
        raise FileNotFoundError(f"Base environment file not found at {root}")
    
def get_env_runner_path(base_path: str) -> str:
    """
    Return the path to the env_runner.py file.
    """
    root = os.path.join(base_path, "runners", "env_runner.py")
    if os.path.isfile(root):
        return root
    else:
        raise FileNotFoundError(f"Env runner file not found at {root}")
    
def get_panda_urdf_path(base_path: str, enable_gripper: bool = False) -> str:
    """
    Return the path to the panda.txt URDF file.
    """
    if enable_gripper:
        root = os.path.join(base_path, "prompts", "robots", "pandaFingers.txt")
    else:
        root = os.path.join(base_path, "prompts", "robots", "panda.txt")
    if os.path.isfile(root):
        return root
    else:
        raise FileNotFoundError(f"Panda URDF file not found at {root}")


def get_universal_attachment_paths(base_path: str, options: dict) -> list[str]:
    """
    Return all file paths under prompts/attachments.
    """
    enable_gripper = options.get('enable_gripper', False)
        
    paths = []

    paths.append(get_base_env_path(base_path))
    paths.append(get_env_runner_path(base_path))
    paths.append(get_panda_urdf_path(base_path, enable_gripper))
    return paths

def get_task_attachment_paths(base_path: str, task_name: str) -> list[str]:
    """
    Return all file paths under prompts/tasks/{task_name}/attachments.
    """
    paths = []
    root = os.path.join(base_path, "envs", task_name)
    if os.path.isdir(root):
        for fn in os.listdir(root):
            fp = os.path.join(root, fn)
            if os.path.isfile(fp):
                paths.append(fp)
    return paths

def get_observer_attachment_paths(base_path: str, timestamp: str):
    json_path = Path(os.path.join(base_path, "logs", timestamp, f"top_runs"))
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found")
    
    attachment_path = []
    for folder in os.listdir(json_path):
        folder_path = os.path.join(json_path, folder)
        video_path = os.path.join(folder_path, "rollout.mp4")
        # extract first frame and save alongside
        if os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(folder_path, "first_frame.png")
                cv2.imwrite(frame_path, frame)
                attachment_path.append(frame_path)
            cap.release()

        # run_json_path = os.path.join(folder_path, f"run.json")
        # attachment_path.append(run_json_path)
    return attachment_path

def get_result_log_json(base_path: str, timestamp: str) -> list[dict]:
    logs_dir = os.path.join(base_path, "logs", timestamp, "top_runs")
    log_entries = []
    for tool_dir in os.listdir(logs_dir):
        run_file = os.path.join(logs_dir, tool_dir, "run.json")
        with open(run_file, "r") as f:
            log_entries.append(json.load(f))
    return log_entries

def get_result_video(base_path: str, timestamp: str) -> list[tuple[str, bytes]]:
    """
    Return all video contents under logs/{timestamp}.
    """
    video_contents = []
    logs_dir = os.path.join(base_path, "logs", timestamp)
    for run_dir in os.listdir(logs_dir):
        folder = os.path.join(logs_dir, run_dir)
        vid_path = os.path.join(folder, "rollout.mp4")
        if os.path.isfile(vid_path):
            with open(vid_path, "rb") as vf:
                video_contents.append((
                    os.path.basename(run_dir),
                    vf.read(),
                    "video/mp4"
                ))
    return video_contents

def get_log_results_inline(client,base_path: str, timestamp: str) -> list[any]:
    """
    Return all screenshot and urdf contents under logs/{timestamp}.
    """
    log_json = get_result_log_json(base_path, timestamp)

    imgs = []
    urdfs = []

    logs_dir = os.path.join(base_path, "logs", timestamp)
    for idx, run_dir in enumerate(os.listdir(logs_dir)):
        folder = os.path.join(logs_dir, run_dir)
        img_path = os.path.join(folder, "screenshot.png")

        # parts.append(f"Design {idx}")
        # parts.append(f"Design Name: {run_dir}")
        # parts.append(f"URDF: {log_json[idx]['full_urdf']}")

        while True:
            try:
                img = client.files.upload(file=img_path)
                break # Success, exit loop
            except Exception as e:
                print(f"Error uploading screenshot {img_path} for {run_dir}: {e}. Retrying in 5 seconds...")
                time.sleep(1)

        imgs.append(img)
        urdfs.append(log_json[idx]['full_urdf'])
        print(run_dir)

        # with open(img_path, 'rb') as f:
        #     img_bytes = f.read()
        #     imgs.append(
        #         types.Part.from_bytes(
        #             data=img_bytes,
        #             mime_type='image/png'
        #         )
        #     )

    return imgs, urdfs

def upload_attachments(client, file_paths: list[str]) -> list:
    """
    Upload given file paths via client.files.upload and return the upload objects.
    Handle potential API connection errors during upload.
    """
    attached = []
    type_list = []
    for fp in file_paths:
        while True: # Start retry loop
            try:
                if Path(fp).suffix == ".mp4" or Path(fp).suffix == ".png":
                    upload_result = client.files.upload(file=fp)
                else:
                    mime, _ = mimetypes.guess_type(fp)
                    with open(fp, "rb") as f:
                        data = io.BytesIO(f.read())
                    upload_result = client.files.upload(file=data, config=dict(mime_type=mime))
                
                # If upload succeeded:
                attached.append(upload_result)
                # Determine type for logging (can be done after successful upload)
                if Path(fp).suffix == ".mp4" or Path(fp).suffix == ".png":
                    type_list.append(Path(fp).suffix)
                else:
                    mime, _ = mimetypes.guess_type(fp) # Guess again or store from above
                    type_list.append(mime)
                
                break # Exit retry loop on success
            
            except Exception as e:  # Catching a broad exception for network/API errors
                print(f"Error uploading file '{fp}': {e}. Retrying in 1 seconds...")
                time.sleep(1) # Wait before retrying

    return attached, type_list

def compose_prompt_and_files(client, file_paths, prompt):
    """
    Upload file_paths, then append prompt (either string or list) to the uploads.
    """
    file_attached, type_list = upload_attachments(client, file_paths)

    total = len(file_paths)
    print(f"\nUploaded {total} file{'s' if total != 1 else ''} for the sampling prompt:")
    header = f"{'No.':<4} {'File Path':<100}     {'Type':<20}"
    print(header)
    print('-' * len(header))
    for idx, (fp, t) in enumerate(zip(file_paths, type_list), start=1):
        print(f"{idx:<4} {fp:<100}     {t:<20}")
    print()  # blank line

    return file_attached + prompt if isinstance(prompt, list) else file_attached + [prompt]



