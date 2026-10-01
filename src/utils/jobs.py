import joblib

def save_artifact(artifact, path):
    joblib.dump(artifact, path)

def load_artifact(path):
    return joblib.load(path)