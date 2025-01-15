"""
model.py for Llama (ollama:llama3.2:1b) model using aisuite.
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
logger = logging.getLogger("LlamaModel")


class TritonPythonModel:
    def initialize(self, args):
        self.client = ai.Client()
        self.models = ["ollama:llama3.2:1b"]
        logger.info(f"Llama model(s) configured: {self.models}")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            if input_tensor is None:
                error_msg = "Input tensor 'INPUT' is missing."
                logger.error(error_msg)
                raise ValueError(error_msg)

            input_array = input_tensor.as_numpy()
            if len(input_array.shape) == 1:
                raw_data = input_array[0]
            else:
                raw_data = input_array[0, 0]

            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8")

            # Try to parse as JSON
            messages = []
            try:
                chat_obj = json.loads(raw_data)
                if isinstance(chat_obj, list):
                    messages = chat_obj
                else:
                    messages = [{"role": "user", "content": str(chat_obj)}]
            except:
                messages = [{"role": "user", "content": raw_data}]

            # Ensure we have at least a system prompt
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages.insert(0, {"role": "system", "content": "You are a helpful AI assistant."})

            logger.info(f"Messages for Llama: {messages}")

            for model in self.models:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.75,
                )
                output_content = response.choices[0].message.content
                logger.info(f"Llama response: {output_content}")
                output_tensor = pb_utils.Tensor("OUTPUT", np.array([output_content], dtype=object))
                responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses