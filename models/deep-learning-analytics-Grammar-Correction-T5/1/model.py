import triton_python_backend_utils as pb_utils
import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration


class TritonPythonModel:
    def initialize(self, args):
        """
        Load T5 model and tokenizer from Hugging Face.
        """
        self.model_name = "deep-learning-analytics/GrammarCorrector"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.num_return_sequences = 1
        self.num_beams = 5

    def correct_grammar(self, input_text):
        """
        Ensures 'input_text' is a single string, then runs it through the T5 model.
        """
        # Handle bytes -> str
        if isinstance(input_text, bytes):
            input_text = input_text.decode("utf-8")
        # Handle numpy arrays -> Python list
        elif isinstance(input_text, np.ndarray):
            input_text = input_text.tolist()

        # If we have a list with exactly one element, extract it
        if isinstance(input_text, list) and len(input_text) == 1:
            input_text = input_text[0]

        # Final check: must be a string
        if not isinstance(input_text, str):
            raise ValueError(f"Input must be a single string. Got: {type(input_text)} => {input_text}")

        # Tokenize
        batch = self.tokenizer(
            [input_text],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        translated = self.model.generate(
            **batch,
            max_length=64,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences
        )

        # Decode results
        corrected_texts = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return corrected_texts

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get the tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "DUMMY_INPUT")
            if input_tensor is None:
                raise ValueError("Input tensor 'DUMMY_INPUT' is missing.")

            input_array = input_tensor.as_numpy()

            # We expect a total of 3 dimensions: (batch, dim1, dim2) => e.g. [1,1,1]
            # Triton automatically prepends the batch dimension because max_batch_size>0.
            if len(input_array.shape) != 3:
                raise ValueError(
                    f"Received shape {input_array.shape}. Expected a 3D array, e.g. (1,1,1)."
                )

            # Extract the text from first indices
            input_text = input_array[0, 0, 0]

            # If it's bytes, decode
            if isinstance(input_text, bytes):
                input_text = input_text.decode("utf-8")

            # Correct grammar
            corrected_texts = self.correct_grammar(input_text)

            # Create output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", np.array(corrected_texts, dtype=object))
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses