import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

# Connect to Triton server
client = InferenceServerClient("localhost:8000")


# Dynamic user command for OpenAI
user_command = "Correct this sentence: My name are Usman"

# Create input tensor with batch dimension
inputs = [InferInput("INPUT", [1, 1], "BYTES")]
inputs[0].set_data_from_numpy(np.array([[user_command]], dtype=object))

# Send inference request to OpenAI model
response = client.infer("openai", inputs)
print("Model Response OpenAI:", response.as_numpy("OUTPUT"))

# Now call "Llama" with similar command
response = client.infer("Llama", inputs)
print("Model Response Llama:", response.as_numpy("OUTPUT"))

# Now call "deep-learning-analytics-Grammar-Correction-T5" with shape (1,1,1)
inputs = [InferInput("DUMMY_INPUT", [1, 1, 1], "BYTES")]
inputs[0].set_data_from_numpy(np.array([[["My name are usman"]]], dtype=object))

response = client.infer("deep-learning-analytics-Grammar-Correction-T5", inputs)
print("Model Response Grammar Correction:", response.as_numpy("OUTPUT"))