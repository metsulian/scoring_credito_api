import pandas as pd
import kagglehub
from pathlib import Path

from src.config import DB_PATH

# Baixa o dataset do kaggle
def download_raw_data() -> None:
    file_path = Path(DB_PATH) / 'train.csv'
    
    if file_path.exists():
        print(f"O dataset já está baixado em: {file_path}")
        return

    path: str = kagglehub.dataset_download(
        "parisrohan/credit-score-classification",
        output_dir=str(DB_PATH),
        path='train.csv'
    )
    print(f"Dados guardados em: {path}")

def load_data() -> pd.DataFrame:
    df = pd.read_csv(f"{DB_PATH}/train.csv", low_memory=False)
    return df




