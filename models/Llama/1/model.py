import triton_python_backend_utils as pb_utils
import aisuite as ai
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the AI Suite client and model configuration.
        """
        self.client = ai.Client()
        self.models = ["ollama:llama3.2:1b"]

    def execute(self, requests):
        """
        Process incoming requests and respond dynamically based on the user input command.
        """
        responses = []
        for request in requests:
            # Extract the input tensor named "INPUT"
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            if input_tensor is None:
                raise ValueError("Input tensor 'INPUT' is missing.")

            # Convert input tensor to a single string (command)
            input_array = input_tensor.as_numpy()
            if len(input_array.shape) == 1:
                # Shape is (1,)
                user_command = input_array[0]
            elif len(input_array.shape) == 2:
                # Shape is (batch, 1)
                user_command = input_array[0, 0]
            else:
                raise ValueError(f"Unexpected input shape {input_array.shape}. Expected 1D or 2D array.")

            # Decode bytes to string if necessary
            if isinstance(user_command, bytes):
                user_command = user_command.decode("utf-8")

            # Construct messages for the AI model
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_command},
            ]

            # Send the request to the AI Suite client
            for model in self.models:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.75,
                )
                output_content = response.choices[0].message.content

                # Create an output tensor with the AI's response
                output_tensor = pb_utils.Tensor("OUTPUT", np.array([output_content], dtype=object))
                responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses