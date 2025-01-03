import triton_python_backend_utils as pb_utils
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        # Load the Hugging Face T5 model and tokenizer
        self.model_name = 'deep-learning-analytics/GrammarCorrector'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.num_return_sequences = 2  # Default number of return sequences
        self.num_beams = 5  # Default number of beams for beam search

    def correct_grammar(self, input_text):
        # Tokenize input text
        batch = self.tokenizer(
            [input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt"
        ).to(self.device)

        # Generate corrected grammar outputs
        translated = self.model.generate(
            **batch, max_length=64, num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences, temperature=1.5
        )

        # Decode and return the corrected outputs
        corrected_texts = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return corrected_texts

    def execute(self, requests):
        responses = []
        for request in requests:
            # Assume the input tensor name is "INPUT" and contains the text to be corrected
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_text = input_tensor.as_numpy()[0].decode('utf-8')

            # Correct grammar using the T5 model
            corrected_texts = self.correct_grammar(input_text)

            # Format the corrected texts as output
            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array(corrected_texts, dtype=object)
            )
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses