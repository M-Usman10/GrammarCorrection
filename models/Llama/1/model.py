"""
model.py for Llama (ollama:llama3.2:1b) model using aisuite.

This module routes user commands to the Llama model, retrieving responses
via aisuite and returning them in a Triton-compatible format.

Best Practices:
1. Similar to the OpenAI model, keep an eye on input shapes.
2. Use logging to monitor usage, conversation content, and performance.
3. Implement security measures if exposing publicly.

"""

import triton_python_backend_utils as pb_utils
import aisuite as ai
import numpy as np
import logging

# ------------------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("LlamaModel")


class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the AI Suite client and model configuration.
        """
        self.client = ai.Client()
        self.models = ["ollama:llama3.2:1b"]
        logger.info(f"Llama model(s) configured: {self.models}")

    def execute(self, requests):
        """
        Process incoming requests and respond dynamically based on the user input command.
        """
        responses = []
        for request in requests:
            # Extract the input tensor named "INPUT"
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            if input_tensor is None:
                error_msg = "Input tensor 'INPUT' is missing."
                logger.error(error_msg)
                raise ValueError(error_msg)

            input_array = input_tensor.as_numpy()
            logger.debug(f"Received input array with shape {input_array.shape}")

            # Handle both 1D ([1]) and 2D ([1,1]) shapes
            if len(input_array.shape) == 1:
                user_command = input_array[0]
            elif len(input_array.shape) == 2:
                user_command = input_array[0, 0]
            else:
                error_msg = f"Unexpected input shape {input_array.shape}. Expected 1D or 2D."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Decode bytes if necessary
            if isinstance(user_command, bytes):
                user_command = user_command.decode("utf-8")
            logger.info(f"User command: {user_command}")

            # Construct messages for the Llama model
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_command},
            ]
            logger.debug(f"Messages for Llama model: {messages}")

            # Send the request to the Llama model
            for model in self.models:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.75,
                )
                output_content = response.choices[0].message.content
                logger.info(f"Llama response: {output_content}")

                # Create an output tensor with the AI's response
                output_tensor = pb_utils.Tensor("OUTPUT", np.array([output_content], dtype=object))
                responses.append(pb_utils.InferenceResponse([output_tensor]))

        logger.info("All Llama requests processed successfully.")
        return responses