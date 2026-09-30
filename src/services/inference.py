from fastapi import HTTPException
from fastapi import FastAPI
from src.functions.model import make_pred, load_model
from pydantic import BaseModel, Field
from typing import Literal
from contextlib import asynccontextmanager
from src.config import MODEL_PATH
import logging

logger = logging.getLogger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(MODEL_PATH)
    logger.info("Modelo carregado")
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

@app.get("/health")
def health():
    return {
        "status": "OK"
    }

@app.post("/predict", response_model=PredOutput)
def predict(payload: PredInput) -> PredOutput:
    try:
        return make_pred(app.state.model, payload.model_dump())
    except Exception:
        logger.exception("Erro em /predict")
        raise HTTPException(status_code = 500)