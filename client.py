"""
This is a sample client script demonstrating how to interact with multiple Triton
models (OpenAI, Llama, a T5 grammar corrector, Happy Transformer, and potentially more).
It uses the Triton HTTP client to send inference requests and prints the responses.

Best Practices and Recommendations:
1. Make sure Triton Server is running and accessible at "localhost:8000".
2. Validate input shapes and data types match the model configurations on the server.
3. Consider adding error handling and retry logic for production environments.
4. Log requests and responses for auditing and debugging.
5. Keep client code separate from business logic, so this file only handles calling
   the models, not domain-specific tasks.
"""

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
import logging

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("TritonClient")

# ------------------------------------------------------------------------------
# Connect to Triton Server
# ------------------------------------------------------------------------------
TRITON_SERVER_URL = "localhost:8000"
client = InferenceServerClient(url=TRITON_SERVER_URL)
logger.info(f"Connected to Triton server at {TRITON_SERVER_URL}")

# ------------------------------------------------------------------------------
# Dynamic user command for OpenAI
# ------------------------------------------------------------------------------
user_command_openai = "Correct this sentence: My name are Usman"
logger.info(f"User command for OpenAI model: {user_command_openai}")

# Prepare input for OpenAI model
inputs_openai = [InferInput("INPUT", [1, 1], "BYTES")]
inputs_openai[0].set_data_from_numpy(np.array([[user_command_openai]], dtype=object))
logger.debug(f"OpenAI input tensor shape: {inputs_openai[0].shape()}")

# Send inference request to OpenAI model
response_openai = client.infer("openai", inputs_openai)
logger.info("Inference request sent to OpenAI model.")
openai_result = response_openai.as_numpy("OUTPUT")
logger.info(f"Model Response OpenAI: {openai_result}")

print("Model Response OpenAI:", openai_result)

# ------------------------------------------------------------------------------
# Now call "Llama" model with similar command
# ------------------------------------------------------------------------------
user_command_llama = user_command_openai  # Reusing the same text for demonstration
logger.info(f"User command for Llama model: {user_command_llama}")

# Prepare input for Llama model
inputs_llama = [InferInput("INPUT", [1, 1], "BYTES")]
inputs_llama[0].set_data_from_numpy(np.array([[user_command_llama]], dtype=object))
logger.debug(f"Llama input tensor shape: {inputs_llama[0].shape()}")

# Send inference request to Llama model
response_llama = client.infer("Llama", inputs_llama)
logger.info("Inference request sent to Llama model.")
llama_result = response_llama.as_numpy("OUTPUT")
logger.info(f"Model Response Llama: {llama_result}")

print("Model Response Llama:", llama_result)

# ------------------------------------------------------------------------------
# Now call the T5 Grammar Correction model
# ------------------------------------------------------------------------------
grammar_input_text = "My name are usman"
logger.info(f"Grammar correction input: {grammar_input_text}")

# Create input tensor for Grammar Correction model
inputs_grammar = [InferInput("DUMMY_INPUT", [1, 1, 1], "BYTES")]
inputs_grammar[0].set_data_from_numpy(np.array([[[grammar_input_text]]], dtype=object))
logger.debug(f"Grammar correction input tensor shape: {inputs_grammar[0].shape()}")

# Send inference request to Grammar Correction model
response_grammar = client.infer("deep-learning-analytics-Grammar-Correction-T5", inputs_grammar)
logger.info("Inference request sent to Grammar Correction model.")
grammar_result = response_grammar.as_numpy("OUTPUT")
logger.info(f"Model Response T5 Grammar Correction: {grammar_result}")

print("Model Response Grammar T5 Correction:", grammar_result)

# ------------------------------------------------------------------------------
# Now call the Happy Transformer model
# ------------------------------------------------------------------------------
happy_input_text = "My name are usman"
logger.info(f"Input text for Happy Transformer: {happy_input_text}")

# Create input tensor for Happy Transformer model
inputs_happy = [InferInput("INPUT", [1,1], "BYTES")]  # Match config: single dimension for batch
inputs_happy[0].set_data_from_numpy(np.array([[happy_input_text]], dtype=object))
logger.debug(f"Happy Transformer input tensor shape: {inputs_happy[0].shape()}")

# Send inference request to Happy Transformer model
response_happy = client.infer("happytransformer", inputs_happy)
logger.info("Inference request sent to Happy Transformer model.")
happy_result = response_happy.as_numpy("OUTPUT")
logger.info(f"Model Response Happy Transformer: {happy_result}")

print("Model Response Happy Transformer:", happy_result)

logger.info("All inferences completed successfully.")