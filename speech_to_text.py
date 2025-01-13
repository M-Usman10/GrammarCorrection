import numpy as np
import librosa
from tritonclient.http import InferenceServerClient, InferInput

# Connect to Triton server
client = InferenceServerClient("localhost:8000",network_timeout=500)

# Load and preprocess the audio file
audio_file_path = "/Users/muhammadusman/PycharmProjects/GrammarCorrection/name_2.mp3"
audio_data, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)  # Convert to 16 kHz mono
audio_data = audio_data.astype(np.float32)  # Ensure 32-bit floating-point format

# Create inference input for the audio
inputs = [InferInput("AUDIO_INPUT", [len(audio_data)], "FP32")]
inputs[0].set_data_from_numpy(audio_data)

# Send inference request
response = client.infer("whisper", inputs)
print("Whisper Model Transcription:", response.as_numpy("OUTPUT"))