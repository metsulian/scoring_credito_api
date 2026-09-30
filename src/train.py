from src.functions.database import load_data, download_raw_data
from src.utils.preprocessing import split_dataset, process_features
from src.functions.model import train, validate_model, save_model

from src.config import MODEL_PATH

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info('Verificando se os dados estao baixados...')
download_raw_data()

logging.info('Carregando dados...')
df = load_data()

logging.info('Processando features...')
df = process_features(df)
X_train, X_test, y_train, y_test = split_dataset(df)

logging.info('Treinando modelo...')
model = train(X_train, y_train)

logging.info('Validando modelo...')
val = validate_model(model, X_test, y_test)
print(val)

logging.info('Salvando modelo...')
save_model(model, MODEL_PATH)