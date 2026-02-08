# Transformer

A minimal and modular implementation of a Transformer-based NLP model in Python.  
The project is designed for experimentation with Transformer architectures, training workflows, and evaluation pipelines.

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

Evaluation
python src/evaluation/main.py
