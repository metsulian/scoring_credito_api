from src.services.google_api import llm_explain
from src.utils.jobs import save_artifact
from src.functions.shap import make_explainer, explain_input
from src.utils.jobs import load_artifact
from src.functions.model import make_pred, prepare_input

from src.config import MODEL_PATH, EXPLAINER_PATH

model = load_artifact(MODEL_PATH)
#explainer = make_explainer(model)
explainer = load_artifact(EXPLAINER_PATH)

input = {
    "Total_EMI_per_month": 49.574949,
   "Monthly_Inhand_Salary": 1824.843333,
   "Monthly_Balance": 312.49408,
   "Credit_Mix": "Good",
   "Payment_of_Min_Amount": "No",
   "Payment_Behaviour": "High_spent_Small_value_payments",
   "Num_Bank_Accounts": 3,
   "Num_Credit_Card": 4,
   "Interest_Rate": 3,
   "Delay_from_due_date": 3,
   "Num_Credit_Inquiries": 4.0,
   "Outstanding_Debt": 809.98,
   "Credit_History_Age": "22 Years and 1 Months"
}

input_df = prepare_input(input)
pred = make_pred(model, input_df)

exp = explain_input(explainer, input_df, pred['class'])
print(exp)

#save_artifact(explainer, EXPLAINER_PATH)

print(llm_explain(exp, 0))
