import shap
import pandas as pd
import numpy as np

FEATURE_VARIABLES = [
    'Total_EMI_per_month',
    'Monthly_Inhand_Salary',
    'Monthly_Balance',
    'Credit_Mix',
    'Payment_of_Min_Amount',
    'Payment_Behaviour',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Delay_from_due_date',
    'Num_Credit_Inquiries',
    'Outstanding_Debt',
    'Credit_History_Age_Years',
    'Debt_per_Card',
    'Interest_x_Delay', 
    'Interest_x_Debt', 
    'Debt_to_Income',
    'EMI_to_Salary', 
    'Balance_to_Salary'
]

def make_explainer(model):
    return shap.Explainer(model)

def explain_input(explainer, input: pd.DataFrame, pred, top_n:int =5):
    exp = explainer(input)[0, :, pred]
    idx = np.argsort(np.abs(exp.values))[::-1][:top_n]
    data ={
        "top_features": [exp.feature_names[i] for i in idx],
        "top_valores": exp.values[idx]
    }
    return data