# TCM Large Language Model (pro_TCMLLM)

A Traditional Chinese Medicine (TCM) domain-specific large language model based on ChatGLM-6B with parameter-efficient fine-tuning (P-Tuning v2).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [License](#license)

## 🌟 Overview

This project fine-tunes the ChatGLM-6B model for Traditional Chinese Medicine (TCM) domain tasks using the ShenNong TCM Dataset. It leverages P-Tuning v2 for parameter-efficient fine-tuning and supports DeepSpeed for distributed training.

### Key Technologies

- **Base Model**: ChatGLM-6B
- **Fine-tuning Method**: P-Tuning v2 (Parameter-efficient)
- **Optimization**: DeepSpeed integration
- **Framework**: HuggingFace Transformers, PEFT
- **Experiment Tracking**: Weights & Biases (wandb)

## ✨ Features

- ✅ Parameter-efficient fine-tuning with P-Tuning v2
- ✅ DeepSpeed integration for distributed training
- ✅ Multiple prompt templates (Medical, Alpaca, Literature)
- ✅ Web-based demo interface
- ✅ CLI demo for quick testing
- ✅ Quantization support for reduced memory usage
- ✅ Experiment tracking with wandb

## 📁 Project Structure

```
pro_TCMLLM/
├── BaseModels/
│   ├── ChatGLM-6B/          # ChatGLM-6B model files and scripts
│   │   ├── ptuning/         # P-Tuning v2 training scripts
│   │   │   ├── main.py      # Main training script
│   │   │   ├── arguments.py # Training arguments
│   │   │   ├── train.sh     # Training shell script
│   │   │   ├── ds_train_finetune.sh  # DeepSpeed training script
│   │   │   ├── evaluate.sh  # Evaluation script
│   │   │   └── web_demo.py  # Web demo for fine-tuned model
│   │   ├── web_demo.py      # Web demo for base model
│   │   ├── cli_demo.py      # CLI demo
│   │   └── requirements.txt # Dependencies
│   └── chatglm-6b/          # Model configuration files
├── code/
│   ├── templates/           # Prompt templates
│   │   ├── med_template.json      # Medical/TCM template
│   │   ├── alpaca.json            # Alpaca template
│   │   └── literature_template.json
│   ├── tools/
│   │   └── hf_stable_downloader.py  # HuggingFace model downloader
│   ├── utils/
│   │   └── prompter.py      # Prompt formatting utility
│   └── requirements.txt
├── datas/
│   └── ShenNong_TCM_Dataset/  # TCM training dataset
├── output/                    # Model checkpoints
└── wandb/                     # Experiment logs
```

## 📦 Requirements

### Hardware Requirements

- **Minimum**: NVIDIA GPU with 12GB VRAM
- **Recommended**: NVIDIA GPU with 24GB VRAM (e.g., RTX 4090, A100)
- **For Inference**: 6GB VRAM with quantization

### Software Requirements

- Python 3.8+
- CUDA 11.0+
- PyTorch 1.10+

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd pro_TCMLLM
```

### 2. Create Virtual Environment

```bash
conda create -n tcmllm python=3.8
conda activate tcmllm
```

### 3. Install Dependencies

```bash
# Install base model dependencies
pip install -r BaseModels/ChatGLM-6B/requirements.txt

# Install additional dependencies
pip install -r code/requirements.txt
```

### 4. Download Base Model

```bash
# Option 1: Using the download script
bash BaseModels/download_chatglm6b.sh

# Option 2: Using HuggingFace downloader
python code/tools/hf_stable_downloader.py --model THUDM/chatglm-6b --output BaseModels/chatglm-6b
```

## 🚀 Quick Start

### Run Base Model Demo

```bash
# Web demo
python BaseModels/ChatGLM-6B/web_demo.py

# CLI demo
python BaseModels/ChatGLM-6B/cli_demo.py
```

### Run Fine-tuned Model Demo

```bash
cd BaseModels/ChatGLM-6B/ptuning
bash web_demo.sh
```

## 🎓 Training

### P-Tuning v2 Training

**Single GPU Training:**

```bash
bash BaseModels/ChatGLM-6B/ptuning/train.sh
```

Default configuration:
- Pre-sequence length: 128
- Learning rate: 2e-2
- Max steps: 3000
- Batch size: 4
- Gradient accumulation: 2
- GPU usage: ~80% on RTX 4090 24GB

**Multi-GPU Training with DeepSpeed:**

```bash
cd BaseModels/ChatGLM-6B/ptuning
bash ds_train_finetune.sh
```

### Training Parameters

Key parameters in `train.sh`:

```bash
PRE_SEQ_LEN=128          # P-Tuning prefix length
LR=2e-2                  # Learning rate
--max_source_length 256  # Input sequence length
--max_target_length 256  # Output sequence length
--per_device_train_batch_size 4
--gradient_accumulation_steps 2
--max_steps 3000
--fp16                   # Mixed precision training
```

### Custom Training Data

Prepare your data in JSON format:

```json
[
  {
    "query": "Your question here",
    "response": "Expected answer here"
  }
]
```

Update the training script:

```bash
--train_file datas/your_train.json \
--validation_file datas/your_val.json \
--prompt_column query \
--response_column response
```

## 📊 Evaluation

```bash
bash BaseModels/ChatGLM-6B/ptuning/evaluate.sh
```

For fine-tuned models:

```bash
bash BaseModels/ChatGLM-6B/ptuning/evaluate_finetune.sh
```

## 🌐 Deployment

### Web Demo

```bash
python BaseModels/ChatGLM-6B/web_demo.py
```

Access the Gradio interface at `http://localhost:7860`

### API Server

```bash
python BaseModels/ChatGLM-6B/api.py
```

### Quantization (4-bit/8-bit)

For reduced memory usage:

```python
model = AutoModel.from_pretrained(
    "BaseModels/chatglm-6b",
    trust_remote_code=True
).quantize(4).half().cuda()  # 4-bit quantization
```

## ⚙️ Configuration

### Prompt Templates

Templates are located in `code/templates/`. The medical template is optimized for TCM:

```json
{
  "description": "Template used by Med Instruction Tuning",
  "prompt_input": "下面是一个问题，运用医学知识来正确回答提问.\n### 问题:\n{instruction}\n### 回答:\n",
  "prompt_no_input": "下面是一个问题，运用医学知识来正确回答提问.\n### 问题:\n{instruction}\n### 回答:\n",
  "response_split": "### 回答:"
}
```

### DeepSpeed Configuration

DeepSpeed config is in `BaseModels/ChatGLM-6B/ptuning/deepspeed.json`:

```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

## 📝 Model Arguments

### ModelArguments

- `model_name_or_path`: Path to pretrained model
- `ptuning_checkpoint`: Path to P-Tuning v2 checkpoint
- `quantization_bit`: Quantization bits (4/8)
- `pre_seq_len`: Prefix sequence length
- `prefix_projection`: Use prefix projection

### DataTrainingArguments

- `train_file`: Training data file
- `validation_file`: Validation data file
- `prompt_column`: Column name for prompts
- `response_column`: Column name for responses
- `max_source_length`: Maximum input length (default: 1024)
- `max_target_length`: Maximum output length (default: 128)

## 🔍 Experiment Tracking

This project uses Weights & Biases for experiment tracking. Logs are saved in the `wandb/` directory.

To view experiments:

```bash
wandb login
wandb sync wandb/run-<run-id>
```

## 📈 Performance Tips

1. **GPU Memory Optimization**:
   - Use `--fp16` for mixed precision training
   - Enable gradient checkpointing
   - Adjust batch size and gradient accumulation

2. **Training Speed**:
   - Use DeepSpeed for multi-GPU training
   - Set `--predict_with_generate False` during training
   - Increase `gradient_accumulation_steps` for larger effective batch size

3. **Inference Optimization**:
   - Use quantization (4-bit/8-bit)
   - Enable KV cache
   - Batch multiple requests

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is based on ChatGLM-6B. Please refer to the original license terms.

## 🙏 Acknowledgments

- [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) by THUDM
- [P-Tuning v2](https://github.com/THUDM/P-tuning-v2)
- ShenNong TCM Dataset

## 📞 Contact

For questions and issues, please open an issue in the repository.

---

**Note**: This project is for research and educational purposes. Please ensure proper validation before using in production medical applications.
