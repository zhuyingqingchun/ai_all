# MiniMind Agent Project

This project contains tools and scripts for working with the MiniMind model and dataset.

## Directory Structure

- `minimind/` - MiniMind model and dataset related files
- `agent_test/` - Agent testing scripts
- `download_dataset.py` - Script to download MiniMind dataset
- `download_model.py` - Script to download MiniMind model

## Setup

### Install Dependencies

```bash
# Install required packages
pip install huggingface_hub requests langgraph
```

### Download Dataset

```bash
# Download default dataset
python minimind/download_dataset.py

# Download specific files
python minimind/download_dataset.py --files dpo.jsonl sft_1024.jsonl
```

### Download Model

```bash
# Download Transformers format model
python minimind/download_model.py

# Download PyTorch format model
python minimind/download_model.py --type pytorch

# Specify save directory
python minimind/download_model.py --save_dir ./my_model
```

## Usage

### LangGraph Integration

This project uses LangGraph for agent workflows. For more information, see the [LangGraph documentation](https://langchain-ai.github.io/langgraph/).

### Testing

Use the scripts in `agent_test/` directory for testing agent functionality.

## Notes

- For users in China, you can download files from ModelScope as an alternative to Hugging Face.
- Model and dataset files are stored in the `dataset/` and `MiniMind2/` directories respectively.
