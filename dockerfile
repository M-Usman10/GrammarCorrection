FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install aisuite and openai
RUN pip install aisuite openai
RUN pip install -U openai-whisper
RUN apt update && sudo apt install ffmpeg