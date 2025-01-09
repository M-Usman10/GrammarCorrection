# utils/model_utils.py

import os
import re
import shutil

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
    Convert user-inputted model name (e.g. 'openai:gpt-4')
    to a folder-friendly name (e.g. 'openai-gpt4'):
     1) Replace ':' with '-'
     2) Remove dash if directly followed by digits
    """
    temp = raw_model_name.replace(":", "-")
    # Remove dash if it's directly followed by digits
    safe_name = re.sub(r'-(\d+)', r'\1', temp)
    return safe_name

def create_model(raw_model_name: str, models_dir="./models"):
    """
    Create a new model in ./models/<sanitizedName>/1 with:
      - model.py (uses original name)
      - config.pbtxt
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

            if len(input_array.shape) == 1:
                user_cmd = input_array[0]
            elif len(input_array.shape) == 2:
                user_cmd = input_array[0, 0]
            else:
                raise ValueError(f"Unexpected shape {{input_array.shape}}")

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
    Delete folder ./models/<model_folder_name>.
    """
    path = os.path.join(models_dir, model_folder_name)
    if os.path.exists(path):
        shutil.rmtree(path)