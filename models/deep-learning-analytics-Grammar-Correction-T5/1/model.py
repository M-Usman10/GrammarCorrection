"""
model.py for T5 Grammar Correction model.

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
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging

# ------------------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger("GrammarCorrectionModel")


class TritonPythonModel:
    def initialize(self, args):
        """
        Load T5 model and tokenizer from Hugging Face.
        """
        self.model_name = "deep-learning-analytics/GrammarCorrector"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using model: {self.model_name} on device: {self.device}")

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

        # Model hyperparameters
        self.num_return_sequences = 1
        self.num_beams = 5
        logger.info("Grammar Correction Model initialized with num_return_sequences = 1, num_beams = 5")

    def correct_grammar(self, input_text):
        """
        Ensures 'input_text' is a single string, then runs it through the T5 model to generate corrections.

        :param input_text: str or bytes or np.ndarray representing the text to correct.
        :return: List of corrected strings (usually just one).
        """
        logger.debug(f"Original input_text: {input_text}")

        # Convert bytes -> str
        if isinstance(input_text, bytes):
            input_text = input_text.decode("utf-8")
            logger.debug("Decoded bytes to string.")

        # Convert numpy arrays -> Python list
        elif isinstance(input_text, np.ndarray):
            input_text = input_text.tolist()
            logger.debug("Converted np.ndarray to list.")

        # If we have a list with exactly one element, extract it
        if isinstance(input_text, list) and len(input_text) == 1:
            input_text = input_text[0]
            logger.debug("Extracted single element from list.")

        # Final check: must be a string
        if not isinstance(input_text, str):
            error_msg = f"Input must be a single string. Got: {type(input_text)} => {input_text}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Tokenize the input string
        batch = self.tokenizer(
            [input_text],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).to(self.device)
        logger.debug(f"Tokenized input: {batch}")

        # Generate corrected text
        translated = self.model.generate(
            **batch,
            max_length=64,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences
        )
        logger.debug("Inference complete, decoding results.")

        corrected_texts = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        logger.info(f"Corrected text: {corrected_texts}")
        return corrected_texts

    def execute(self, requests):
        """
        Main execution method for Triton inference requests.
        """
        responses = []
        for request in requests:
            # Extract the input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "DUMMY_INPUT")
            if input_tensor is None:
                error_msg = "Input tensor 'DUMMY_INPUT' is missing."
                logger.error(error_msg)
                raise ValueError(error_msg)

            input_array = input_tensor.as_numpy()
            logger.debug(f"Received array with shape {input_array.shape}")

            # We expect shape (batch, dim1, dim2) e.g. [1,1,1]
            if len(input_array.shape) != 3:
                error_msg = f"Received shape {input_array.shape}. Expected a 3D array (e.g. (1,1,1))."
                logger.error(error_msg)
                raise ValueError(error_msg)

            input_text = input_array[0, 0, 0]
            logger.debug(f"Extracted text from array: {input_text}")

            corrected_texts = self.correct_grammar(input_text)

            # Convert the corrected texts into a suitable output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", np.array(corrected_texts, dtype=object))
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        logger.info("All Grammar Correction requests processed successfully.")
        return responses