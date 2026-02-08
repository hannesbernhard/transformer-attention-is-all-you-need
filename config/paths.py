from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
STORAGE = project_root / "storage"
GPT2_FROM_BPE = STORAGE / "gpt2_from_bpe"
MODEL_CONFIG = project_root / "src" / "config" / "config.yaml"
HYPERPARAMETERS = project_root / "src" / "config" / "hyperparameters.yaml"
BEST_MODELS = STORAGE / "best_models"
EVALUATION = STORAGE / "evaluate"
