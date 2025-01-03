FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Set environment variables
ENV MODEL_NAME=deep-learning-analytics/GrammarCorrector
ENV HF_HOME=/models/huggingface

# Update pip and setuptools to avoid version-related issues
RUN pip install --upgrade pip setuptools

# Install necessary Python packages
RUN pip install --no-cache-dir \
    transformers \
    torch \
    sentencepiece \
    aisuite \
    openai \
    accelerate

# Pre-download the model
RUN python3 -c "\
from transformers import T5Tokenizer, T5ForConditionalGeneration; \
model_name = '${MODEL_NAME}'; \
T5Tokenizer.from_pretrained(model_name); \
T5ForConditionalGeneration.from_pretrained(model_name)"

# Ensure the model and dependencies are available when the container starts
RUN mkdir -p /models/huggingface && chmod -R 777 /models

# Default command (optional, adjust for Triton-specific commands if needed)
CMD ["tritonserver", "--model-repository=/models"]