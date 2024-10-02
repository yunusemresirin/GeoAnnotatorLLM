# GeoAnnotator-LLM
## Purpose
This repository is the trainings-server of the GeoAnnotator (GA) for LLMs. It is used for hosting
and finetuning Large Language Models.

## Requirements
On Linux only

### 1. Hardware Requirements
- **NVIDIA GPU**: Required for efficient fine-tuning. 
  - 8 GB+ VRAM for small models (1-3B parameters)
  - 24 GB+ VRAM for larger models (6B-13B parameters)
- **CPU**: Modern multi-core CPU for handling dataset preprocessing and other tasks.
- **RAM**: 
  - 16 GB recommended for smaller models.
  - 32 GB+ for larger models.
- **Storage**: 
  - 50-200 GB SSD for model checkpoints and datasets.

### 2. Software Requirements
- **Python 3.8 or higher**
- **CUDA Toolkit**: Required for GPU support. Install a version compatible with your PyTorch installation.
- **cuDNN**: NVIDIA's library for deep learning.

### 3. Installing Dependencies
```bash
conda create -n GeoAnnotatorLLM \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y

conda activate GeoAnnotatorLLM

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

pip install fastapi "uvicorn[standard]"
```
or with YAML-file GeoAnnotatorLLM.yml:
```bash
conda env create --file GeoAnnotatorLLM.yml --name GeoAnnotatorLLM
```

We also recommend installing the Host-Client LM Studio to host the models.

### 4. Pre-trained Models
To host fine-tune models, the configuration and quantization of these models are required.
- Place the models in the './models/' directory and use it also for hosting the LLMs
- For finetuning, config-file and tokenizer is needed.

To install a model, run:
```bash
(GeoAnnotatorLLMs) python3 Install-Llama-3-1-8B.py
```