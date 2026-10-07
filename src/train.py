from src.functions.database import load_data, download_raw_data
from src.utils.preprocessing import split_dataset, process_features
from src.utils.jobs import save_artifact
from src.functions.model import train, validate_model
from src.functions.shap import make_explainer

from src.config import MODEL_PATH, EXPLAINER_PATH

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
X_train, X_test, y_train, y_test, id_groups = split_dataset(df)

logging.info('Treinando modelo...')
model = train(X_train, y_train, id_groups)
model = model.best_estimator_

logging.info('Validando modelo...')
val = validate_model(model, X_test, y_test)
logging.info(f'Validacao com resultado: {val}')

logging.info('Salvando modelo...')
save_artifact(model, MODEL_PATH)

logging.info('Treinando explainer...')
explainer = make_explainer(model)

logging.info('Salvando explainer...')
save_artifact(explainer, EXPLAINER_PATH)

