import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

# Connect to Triton server
client = InferenceServerClient("localhost:8000")

# Create dummy input since the model doesn't need input
inputs = [InferInput("DUMMY_INPUT", [1], "BYTES")]
inputs[0].set_data_from_numpy(np.array([""], dtype=object))

# Send inference request
response = client.infer("Llama 3.2:1b", inputs)
print("Model Response Llama 3.2:1b:", response.as_numpy("OUTPUT"))

response = client.infer("openai", inputs)
print("Model Response GPT-4o:", response.as_numpy("OUTPUT"))