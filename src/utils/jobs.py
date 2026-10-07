import joblib
from pathlib import Path

def save_artifact(artifact, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)

def load_artifact(path):
    return joblib.load(path)