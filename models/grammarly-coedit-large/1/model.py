"""
model.py for Grammarly CoEdit large model.

This module uses the Python backend for Triton to load a T5 model from Hugging Face,
preprocess input, run inference, and return the corrected sentences.

Best Practices:
1. Validate model architecture matches the config.pbtxt dimensions.
2. Thoroughly handle input and output shapes, raising clear errors on mismatches.
3. Add logging for debug and trace in production scenarios.
"""

import triton_python_backend_utils as pb_utils
import torch
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
import logging

# ------------------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("CoEditlargeModel")


class TritonPythonModel:
    def initialize(self, args):
        """
        Load Grammarly CoEdit large model and tokenizer.
        """
        self.model_name = "grammarly/coedit-large"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using model: {self.model_name} on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def correct_grammar(self, input_text):
        """
        Runs input text through the Grammarly CoEdit large model to generate corrections.

        :param input_text: str or bytes or np.ndarray representing the text to correct.
        :return: List of corrected strings (usually just one).
        """
        logger.debug(f"Original input_text: {input_text}")

        if isinstance(input_text, bytes):
            input_text = input_text.decode("utf-8")
        elif isinstance(input_text, np.ndarray):
            input_text = input_text.tolist()
        if isinstance(input_text, list) and len(input_text) == 1:
            input_text = input_text[0]
        if not isinstance(input_text, str):
            raise ValueError(f"Input must be a single string. Got: {type(input_text)} => {input_text}")

        # Tokenize the input string
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        # Generate corrected text
        outputs = self.model.generate(input_ids, max_length=256)
        corrected_texts = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Corrected text: {corrected_texts}")
        return [corrected_texts]

    def execute(self, requests):
        """
        Main execution method for Triton inference requests.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            if input_tensor is None:
                raise ValueError("Input tensor 'INPUT' is missing.")

            input_text = input_tensor.as_numpy()[0].decode("utf-8")
            corrected_texts = self.correct_grammar(input_text)
            output_tensor = pb_utils.Tensor("OUTPUT", np.array(corrected_texts, dtype=object))
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses