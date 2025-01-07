# utils.py
import os
import re
import shutil
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
        env_map = {}
        raw_dict = dotenv_values(env_file)
        for k, v in raw_dict.items():
            if v is None:
                continue
            # Remove any surrounding quotes
            clean_val = v.strip('\'"')
            env_map[k] = clean_val
        return env_map
    return {}

def list_models(models_dir="./models"):
    """
    Return a list of existing model folder names in ./models/.
    """
    if not os.path.exists(models_dir):
        return []
    return sorted([
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d)) and not d.startswith('.')
    ])

def sanitize_model_name(raw_model_name: str) -> str:
    """
    Convert user-inputted model name (e.g. 'openai:gpt-4') to a folder-friendly name (e.g. 'openai-gpt4'):
    1) Replace ':' with '-'
    2) Remove a dash if it is immediately followed by digits, e.g. '-4' => '4'
    """
    temp = raw_model_name.replace(":", "-")
    # Remove dash if it's directly followed by digits
    safe_name = re.sub(r'-(\d+)', r'\1', temp)
    return safe_name

def create_model(raw_model_name: str, models_dir="./models"):
    """
    Create a new model in ./models/<sanitizedName>/1 with:
      - model.py  (contains the *original* name with ':')
      - config.pbtxt (with sanitized name).
    """
    safe_name = sanitize_model_name(raw_model_name)

    base_path = os.path.join(models_dir, safe_name)
    version_path = os.path.join(base_path, "1")
    model_py_path = os.path.join(version_path, "model.py")
    config_path = os.path.join(base_path, "config.pbtxt")

    os.makedirs(version_path, exist_ok=True)

    model_py_content = f'''\
"""
model.py for {raw_model_name} using aisuite.

This Python backend relay uses aisuite to connect to the model: {raw_model_name}
"""
import triton_python_backend_utils as pb_utils
import aisuite as ai
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("{safe_name}_logger")

class TritonPythonModel:
    def initialize(self, args):
        # Keep the original name in the list
        self.client = ai.Client()
        self.models = ["{raw_model_name}"]
        logger.info(f"Model(s) configured: {{self.models}}")

    def execute(self, requests):
        responses = []
        for req in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(req, "INPUT")
            if input_tensor is None:
                raise ValueError("Input tensor 'INPUT' is missing.")
            input_array = input_tensor.as_numpy()

            # Accept shape [1] or [1,1]
            if len(input_array.shape) == 1:
                user_cmd = input_array[0]
            elif len(input_array.shape) == 2:
                user_cmd = input_array[0, 0]
            else:
                raise ValueError(f"Unexpected shape {{input_array.shape}} (expected 1D or 2D).")

            if isinstance(user_cmd, bytes):
                user_cmd = user_cmd.decode("utf-8")
            logger.info(f"User command: {{user_cmd}}")

            messages = [
                {{"role": "system", "content": "You are a helpful AI assistant."}},
                {{"role": "user", "content": user_cmd}}
            ]

            for model in self.models:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.75,
                )
                out = response.choices[0].message.content
                logger.info(f"Response from {{model}}: {{out}}")
                output_tensor = pb_utils.Tensor("OUTPUT", np.array([out], dtype=object))
                responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses
'''

    config_content = f'''\
name: "{safe_name}"
backend: "python"

max_batch_size: 8

input [
  {{
    name: "INPUT"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }}
]

output [
  {{
    name: "OUTPUT"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }}
]
'''
    with open(model_py_path, 'w', encoding='utf-8') as f:
        f.write(model_py_content)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

def delete_model(model_folder_name, models_dir="./models"):
    """
    Delete the folder ./models/<model_folder_name>.
    """
    path = os.path.join(models_dir, model_folder_name)
    if os.path.exists(path):
        shutil.rmtree(path)

# ------------------- Triton Server Management -------------------

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
    If success == False, we also provide logs for direct display in the UI.
    """
    # Ensure no existing container is running
    if not kill_triton_server():
        return False, "Failed to kill existing Triton server container.", None

    env_vars = load_env_vars(".env")  # dict of {KEY: VAL}
    cmd = [
        "docker", "run",
        # Removed --rm so we can fetch logs on failure
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
    Return more complete Triton server logs from container.
    Using --since=0 to retrieve all logs from container start.
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