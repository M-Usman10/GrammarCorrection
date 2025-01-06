import triton_python_backend_utils as pb_utils
from transformers import pipeline
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        # Initialize Whisper model using Hugging Face Transformers
        self.transcription_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Assuming each request contains audio data
            audio_input = pb_utils.get_input_tensor_by_name(request, "AUDIO_INPUT").as_numpy()

            # Perform speech-to-text using Whisper
            transcription = self.transcription_pipeline(audio_input)["text"]

            # Wrap the transcription into the output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", np.array([transcription], dtype=object))
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses