# This Dockerfile builds a custom Triton server image with pre-installed
# dependencies and pre-downloaded models for grammar correction, including:
# Grammarly CoEdit, deep-learning-analytics T5 model, and HappyTransformer.

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
# Upgrade pip and install Python dependencies
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel
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
# Pre-download models
# ---------------------------------------------------------------------------
RUN python3 <<EOF
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from happytransformer import HappyTextToText, TTSettings
import time

def download_model(model_name):
    """Download model with retries for robustness."""
    for attempt in range(3):
        try:
            print(f"Attempt {attempt + 1}: Downloading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            print(f"{model_name} downloaded successfully.")
            return
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            time.sleep(10)
    raise RuntimeError(f"Failed to download {model_name} after multiple attempts.")

# Download Grammarly CoEdit large model
download_model("grammarly/coedit-large")

# Pre-download deep-learning-analytics T5 model
print('Pre-downloading deep-learning-analytics T5 model...')
t5_tokenizer = T5Tokenizer.from_pretrained('${MODEL_NAME}')
t5_model = T5ForConditionalGeneration.from_pretrained('${MODEL_NAME}')

# Pre-download HappyTransformer model
print('Pre-downloading HappyTransformer model...')
happy_tt = HappyTextToText('T5', '${HAPPYTRANSFORMER_MODEL_ID}')
settings = TTSettings(num_beams=5, min_length=1)
_ = happy_tt.generate_text('grammar: Test sentence', args=settings)

print('Pre-download completed successfully.')
EOF

# ---------------------------------------------------------------------------
# Create Hugging Face cache directory
# ---------------------------------------------------------------------------
RUN mkdir -p /huggingface && chmod -R 777 /huggingface

# ---------------------------------------------------------------------------
# Default command
# ---------------------------------------------------------------------------
CMD ["tritonserver", "--model-repository=/models"]