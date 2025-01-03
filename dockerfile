# This Dockerfile builds a custom Triton server image with pre-installed
# dependencies and a pre-downloaded T5 model for grammar correction,
# as well as GECToR and HappyTransformer models.
#
# Best Practices:
# 1. Use multi-stage builds to reduce final image size.
# 2. Keep instructions in a logical order for minimal rebuild steps.
# 3. Use stable base images for reproducibility.

# Use the NVIDIA Triton server base image
FROM nvcr.io/nvidia/tritonserver:23.10-py3

# ---------------------------------------------------------------------------
# Set environment variables
# ---------------------------------------------------------------------------
ENV MODEL_NAME=deep-learning-analytics/GrammarCorrector
ENV HF_HOME=/huggingface
ENV HAPPYTRANSFORMER_MODEL_ID=vennify/t5-base-grammar-correction

# ---------------------------------------------------------------------------
# Install system packages and dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Upgrade pip and install compatible setuptools and wheel
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Install Python packages (excluding Spacy and GECToR dependencies)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    numpy \
    scikit-learn \
    overrides \
    python-Levenshtein \
    transformers \
    torch \
    sentencepiece \
    aisuite \
    openai \
    accelerate \
    happytransformer

# ---------------------------------------------------------------------------
# Pre-download the T5 and HappyTransformer models
# ---------------------------------------------------------------------------
RUN python3 -c "\
import torch;\
import transformers;\
from transformers import T5Tokenizer, T5ForConditionalGeneration;\
from happytransformer import HappyTextToText, TTSettings;\
import os;\
\
print('Pre-downloading deep-learning-analytics T5 model...');\
t5_tokenizer = T5Tokenizer.from_pretrained('${MODEL_NAME}');\
t5_model = T5ForConditionalGeneration.from_pretrained('${MODEL_NAME}');\
\
print('Pre-downloading HappyTransformer model...');\
happy_tt = HappyTextToText('T5', '${HAPPYTRANSFORMER_MODEL_ID}');\
settings = TTSettings(num_beams=5, min_length=1);\
_ = happy_tt.generate_text('grammar: Test sentence', args=settings);\
\
print('Pre-download completed successfully.')"

# ---------------------------------------------------------------------------
# Ensure the model and dependencies are available when the container starts
# ---------------------------------------------------------------------------
RUN mkdir -p /huggingface && chmod -R 777 /huggingface

# ---------------------------------------------------------------------------
# Default command
# ---------------------------------------------------------------------------
CMD ["tritonserver", "--model-repository=/models"]