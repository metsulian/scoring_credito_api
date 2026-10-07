from src.services.google_api import llm_explain
from src.functions.model import prepare_input, make_pred
from src.utils.jobs import load_artifact
from fastapi import HTTPException, FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from contextlib import asynccontextmanager
from src.functions.shap import explain_input

from src.config import MODEL_PATH, EXPLAINER_PATH
import logging

logger = logging.getLogger(__name__)

MODELS = [
    "gemini-3.5-flash", 
    "gemini-3.1-flash-lite", 
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_artifact(MODEL_PATH)
    logger.info("Modelo carregado")
    app.state.explainer = load_artifact(EXPLAINER_PATH)
    yield

app = FastAPI(
    title="Analise Credito API",
    lifespan=lifespan
)

class PredInput(BaseModel):
    Total_EMI_per_month: float = Field(ge=0)
    Monthly_Inhand_Salary: float = Field(ge=0)
    Monthly_Balance: float = Field(ge=0)
    Credit_Mix: Literal["Bad", "Good", "Standard"] 
    Payment_of_Min_Amount: Literal["No", "Yes"]
    Payment_Behaviour: Literal[
        "High_spent_Large_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments"
    ]
    Num_Bank_Accounts: int = Field(ge=0)
    Num_Credit_Card: int = Field(ge=0)
    Interest_Rate: float = Field(ge=0)
    Delay_from_due_date: float = Field(ge=0)
    Num_Credit_Inquiries: float = Field(ge=0)
    Outstanding_Debt: float = Field(ge=0)
    Credit_History_Age: str

class PredOutput(BaseModel):
    prediction: Literal["Good", "Poor", "Standard"]
    probabilities: dict

class ExplainOutput(BaseModel):
    prediction: Literal["Good", "Poor", "Standard"] | None
    class_: int | None
    probabilities: dict | None
    llm_response: str | None

@app.get("/health")
def health():
    return {
        "status": "OK"
    }

@app.post("/predict", response_model=PredOutput)
def predict(payload: PredInput) -> PredOutput:
    try:
        input = prepare_input(payload.model_dump())
        return make_pred(app.state.model, input)
    except Exception as e:
        logger.exception("Erro em /predict")
        raise HTTPException(status_code = 500) from e

@app.post("/explain", response_model=ExplainOutput)
def explain(payload: PredInput, class_: int, top_n:int = 5):
    input = prepare_input(payload.model_dump())
    data = explain_input(app.state.explainer, input, class_, top_n)
    for model in MODELS:
        try:
                llm_response = llm_explain(data, class_, model)
                return {
                    "prediction": None,
                    "class_": None,
                    "probabilities": None,
                    "llm_response": llm_response
                }
        except Exception as e:
            logger.exception("Erro em exmplain")
            raise HTTPException(status_code=500) from e

@app.post("/predict-explain", response_model=ExplainOutput)
def predict_explain(payload: PredInput):
    input = prepare_input(payload.model_dump())
    pred = make_pred(app.state.model, input)
    pred_class = pred["class_"]
    data = explain_input(app.state.explainer, input, pred_class)
    for model in MODELS:
        try:
            llm_response = llm_explain(data, pred_class, model)
            return {
                "prediction": pred["prediction"],
                "class_": pred_class,
                "probabilities": pred["probabilities"],
                "llm_response": llm_response
            }
        except Exception as e:
            logger.exception("Erro em predict-explain")
            raise HTTPException(status_code=500) from e

