# Transformer

A minimal and modular implementation of a Transformer-based NLP model in Python.  
The project is designed for experimentation with Transformer architectures, training workflows, and evaluation pipelines. Additionally, a small RoPE experiment was conducted and evaluated with different max lengths.

## Requirements

- Python 3.12+
- Poetry (recommended) or pip

## Installation

```bash
git clone <repository-url>
cd transformer
poetry install

or

pip install -r requirements.txt


Configuration
Model, training, and runtime settings are defined via YAML files:
	•	config.yaml for dataset and runtime configuration
	•	hyperparameters.yaml for model architecture and training parameters

Training
python src/run/main.py

set --fetch-data-online if download should be done via execution. If not set, downloaded dataset has to be provided under src/hf_cache

Evaluation
python src/evaluation/main.py

set --fetch-data-online if download should be done via execution. If not set, downloaded dataset has to be provided under src/hf_cache
set --max_len to test different max lengths during evaluation, default is 64, which was used during the training as can be seen in the config file.
