"""
Grammarly CoEdit Grammar Correction model using "grammarly/coedit-large"
with the Python backend for Triton.

This script loads a T5 model via the Transformers library and uses it
to correct grammar for the provided input text.
"""

import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import logging

# ------------------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("CoEditModel")


class TritonPythonModel:
    def initialize(self, args):
        """
        Load the Grammarly CoEdit model and tokenizer.
        """
        logger.info("Initializing Grammarly CoEdit model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
        self.model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large").to(self.device)
        logger.info("Grammarly CoEdit model loaded successfully.")

    def execute(self, requests):
        """
        Process incoming requests from Triton. The user text is expected
        to be provided as a 1D or 2D tensor named "INPUT".
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            if not input_tensor:
                raise ValueError("Input tensor 'INPUT' is missing for CoEdit model.")

            # Convert to numpy array
            input_array = input_tensor.as_numpy()
            logger.debug(f"Received input array of shape {input_array.shape}")

            # Handle 1D ([1]) or 2D ([batch, 1]) shapes
            if len(input_array.shape) == 1:
                text_to_correct = input_array[0]
            elif len(input_array.shape) == 2:
                text_to_correct = input_array[0, 0]
            else:
                raise ValueError(f"Unexpected shape {input_array.shape}. Expected 1D or 2D.")

            # Decode bytes if needed
            if isinstance(text_to_correct, bytes):
                text_to_correct = text_to_correct.decode("utf-8")

            logger.info(f"Text to correct (CoEdit): {text_to_correct}")

            # Tokenize the input
            input_ids = self.tokenizer(
                text_to_correct,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            ).input_ids.to(self.device)

            # Generate the corrected output
            outputs = self.model.generate(input_ids, max_length=256, num_beams=5, num_return_sequences=1)
            corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Corrected text: {corrected_text}")

            # Create Triton output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", np.array([corrected_text], dtype=object))
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses