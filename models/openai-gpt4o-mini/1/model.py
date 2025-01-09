"""
model.py for openai:gpt-4o-mini using aisuite.

This Python backend relay uses aisuite to connect to the model: openai:gpt-4o-mini
"""
import triton_python_backend_utils as pb_utils
import aisuite as ai
import numpy as np
import logging

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

            if len(input_array.shape) == 1:
                user_cmd = input_array[0]
            elif len(input_array.shape) == 2:
                user_cmd = input_array[0, 0]
            else:
                raise ValueError(f"Unexpected shape {input_array.shape}")

            if isinstance(user_cmd, bytes):
                user_cmd = user_cmd.decode("utf-8")
            logger.info(f"User command: {user_cmd}")

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_cmd}
            ]

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
