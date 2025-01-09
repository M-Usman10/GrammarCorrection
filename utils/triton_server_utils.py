# utils/triton_server_utils.py

import os
import subprocess
import time

from dotenv import dotenv_values

TRITON_CONTAINER_NAME = "custom_tritonserver"

def load_env_vars(env_file=".env"):
    """
    Loads environment variables from .env using python-dotenv.
    Returns a dict of {VAR_NAME: value}.
    """
    if os.path.exists(env_file):
        import re
        from dotenv import dotenv_values
        env_map = {}
        raw_dict = dotenv_values(env_file)
        for k, v in raw_dict.items():
            if v is None:
                continue
            clean_val = v.strip('\'"')
            env_map[k] = clean_val
        return env_map
    return {}

def get_container_status(container_name):
    """
    Return the Docker container's state string if it's running, otherwise ''.
    """
    try:
        output = subprocess.check_output([
            "docker", "ps", "--filter", f"name={container_name}",
            "--format", "{{.State}}"
        ])
        state_str = output.decode().strip()
        if state_str == "running":
            return "running"
        return ""
    except subprocess.CalledProcessError:
        return ""
    except FileNotFoundError:
        return ""

def kill_triton_server():
    """
    Forcefully kill the Triton Docker container if it's running.
    Returns True if killed successfully or was not running, False otherwise.
    """
    status = get_container_status(TRITON_CONTAINER_NAME)
    if status == "running":
        try:
            subprocess.check_output(["docker", "kill", TRITON_CONTAINER_NAME])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error killing container: {e.output.decode('utf-8')}")
            return False
    return True

def start_triton_server():
    """
    Start the Triton server with environment variables from .env,
    passing them via -e KEY=VALUE to docker.
    Returns (success: bool, msg: str, logs: str or None).
    If success == False, logs may be provided.
    """
    # Ensure no existing container is running
    if not kill_triton_server():
        return False, "Failed to kill existing Triton server container.", None

    env_vars = load_env_vars(".env")  # dict of {KEY: VAL}
    cmd = [
        "docker", "run",
        "--rm",
        "--name", TRITON_CONTAINER_NAME,
        "-p", "8000:8000",
        "-v", f"{os.getcwd()}/models:/models",
        "-v", f"{os.getcwd()}/huggingface:/huggingface",
    ]
    # Add environment variables from .env
    for k, v in env_vars.items():
        cmd.extend(["-e", f"{k}={v}"])

    # The rest of the command
    cmd.extend([
        "tritonserver-custom",
        "tritonserver", "--model-repository=/models", "--disable-auto-complete-config"
    ])

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Let container attempt to start
        if get_container_status(TRITON_CONTAINER_NAME) == "running":
            return True, "Triton server started successfully.", None
        else:
            # If not running, fetch logs
            try:
                logs = subprocess.check_output(
                    ["docker", "logs", TRITON_CONTAINER_NAME],
                    stderr=subprocess.STDOUT
                ).decode('utf-8', errors='replace')
            except Exception as e:
                logs = f"Error retrieving logs: {e}"
            return False, "Failed to start Triton server.", logs
    except Exception as e:
        return False, str(e), None

def stop_triton_server():
    """
    Stop the Docker container if running.
    Returns (success, message).
    """
    status = get_container_status(TRITON_CONTAINER_NAME)
    if status == "running":
        try:
            subprocess.check_output(["docker", "stop", TRITON_CONTAINER_NAME])
            return True, "Triton server stopped."
        except Exception as e:
            return False, str(e)
    else:
        return True, "Triton server is not running."

def get_triton_logs():
    """
    Return complete Triton server logs from container.
    """
    status = get_container_status(TRITON_CONTAINER_NAME)
    if status == "running":
        try:
            logs = subprocess.check_output([
                "docker", "logs", "--since=0",
                TRITON_CONTAINER_NAME
            ], stderr=subprocess.STDOUT)
            return logs.decode("utf-8", errors="replace")
        except subprocess.CalledProcessError as e:
            return f"Error retrieving logs: {e.output.decode('utf-8', errors='replace')}"
        except Exception as e:
            return f"Error retrieving logs: {e}"
    else:
        return "Triton server not running."