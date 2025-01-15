"""
model.py for openai:gpt-4o-mini using aisuite.
"""
import triton_python_backend_utils as pb_utils
import aisuite as ai
import numpy as np
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("openai-gpt4o-mini_logger")

class TritonPythonModel:
    def initialize(self, args):
        self.client = ai.Client()
        self.models = ["openai:gpt-4o-mini"]
        logger.info(f"Model(s) configured: {self.models}")

    def execute(self, requests):
        responses = []
        for req in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(req, "INPUT")
            if input_tensor is None:
                raise ValueError("Input tensor 'INPUT' is missing.")
            input_array = input_tensor.as_numpy()

            # Expect shape [1,1] or [1]
            if len(input_array.shape) == 1:
                raw_data = input_array[0]
            else:
                raw_data = input_array[0, 0]

            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8")

            # Attempt to parse as JSON
            # If it fails, assume it's a single string prompt (grammar correction).
            messages = []
            try:
                chat_obj = json.loads(raw_data)
                if isinstance(chat_obj, list):
                    # We have a list of messages
                    messages = chat_obj
                    # Ensure there's at least a system + user
                    # We'll just pass as is
                else:
                    # If it's not a list, treat it as plain text
                    messages = [{"role": "user", "content": str(chat_obj)}]
            except:
                # raw_data is just a single string
                messages = [{"role": "user", "content": raw_data}]

            # If we do not see a system in messages, we can prepend a default
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages.insert(0, {"role": "system", "content": "You are a helpful AI"})

            logger.info(f"User command: {raw_data}")
            logger.info(f"Passing messages: {messages}")

            for model in self.models:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.75,
                )
                out = response.choices[0].message.content
                logger.info(f"Response from {model}: {out}")
                output_tensor = pb_utils.Tensor("OUTPUT", np.array([out], dtype=object))
                responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses