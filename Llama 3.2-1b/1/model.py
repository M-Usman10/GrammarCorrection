import triton_python_backend_utils as pb_utils
import aisuite as ai
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        # Initialize AI Suite client
        self.client = ai.Client()
        self.models = ["ollama:llama3.2:1b"]

    def execute(self, requests):
        responses = []
        for request in requests:
            for model in self.models:
                messages = [
                    {"role": "system", "content": "Respond in Pirate English."},
                    {"role": "user", "content": "Tell me a joke."},
                ]
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.75,
                )
                joke = response.choices[0].message.content
                output_tensor = pb_utils.Tensor("OUTPUT", np.array([joke], dtype=object))
                responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses