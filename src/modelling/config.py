from pathlib import Path

# [3] means go up 3 levels from the current file 
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRPATH = str(PROJECT_ROOT / "abalone.csv")
MODELS_DIRPATH = str(PROJECT_ROOT / "models")
