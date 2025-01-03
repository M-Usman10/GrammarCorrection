"""
Happy Transformer Grammar Correction model using "vennify/t5-base-grammar-correction"
with the Python backend for Triton.

This script loads a T5 model via the HappyTransformer library and uses it
to correct grammar for the provided input text.
"""

import triton_python_backend_utils as pb_utils
import numpy as np
import logging

# Happy Transformer modules
from happytransformer import HappyTextToText, TTSettings

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("HappyTransformerModel")

class TritonPythonModel:
    def initialize(self, args):
        """
        Load the T5-based grammar correction model (HappyTransformer).
        """
        logger.info("Initializing Happy Transformer model...")
        # Model name from Happy Transformer: vennify/t5-base-grammar-correction
        self.happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
        # Use recommended generation settings
        self.args = TTSettings(num_beams=5, min_length=1)
        logger.info("HappyTransformer model loaded successfully.")

    def execute(self, requests):
        """
        Process incoming requests from Triton. The user text is expected
        to be provided as a 1D or 2D tensor named "INPUT".
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            if not input_tensor:
                raise ValueError("Input tensor 'INPUT' is missing for HappyTransformer model.")

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

            logger.info(f"Text to correct (HappyTransformer): {text_to_correct}")

            # Add "grammar: " prefix as required by the model
            # per docs: "Add the prefix 'grammar: ' before each input"
            prefixed_text = f"grammar: {text_to_correct}"

            # Generate correction
            result = self.happy_tt.generate_text(prefixed_text, args=self.args)
            logger.info(f"Corrected text: {result.text}")

            # Create Triton output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", np.array([result.text], dtype=object))
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses