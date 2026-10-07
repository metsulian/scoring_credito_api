import pandas as pd
import xgboost as xgb
import re

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedGroupKFold

from src.config import MODEL_PATH

PARAM_GRID = {
    'n_estimators': [500, 700],
    'max_depth': [6, 9],
    'learning_rate': [0.05],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [5, 10],
    'subsample': [0.7, 0.9]
}

MODEL = xgb.XGBClassifier(
        random_state=4,
        num_class=3,
        objective='multi:softprob',
        eval_metric='mlogloss'
    )

def validate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    auc_macro = roc_auc_score(
    y_test,
    y_proba,
    multi_class='ovr',
    average='macro'
    )

    auc_weighted = roc_auc_score(
        y_test,
        y_proba,
        multi_class='ovr',
        average='weighted'
    )

    gini_macro = 2 * auc_macro - 1
    gini_weighted = 2 * auc_weighted - 1

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_cv = float(model.score(X_test, y_test))

    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_cv": f1_cv,
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "gini_macro": gini_macro,
        "gini_weighted": gini_weighted
    }

def train(
    X_train, y_train, id_groups, cv: int = 5
):
    random_search = RandomizedSearchCV(
        estimator=MODEL,
        param_distributions=PARAM_GRID,
        scoring='f1_macro',
        verbose=0,
        cv = StratifiedGroupKFold(n_splits=cv),
        random_state=4,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train, groups=id_groups)

    return random_search

def prepare_input(
    artifact: dict
) -> pd.DataFrame:
    credit_mix = artifact['Credit_Mix'] # Categorico
    payment_min = artifact['Payment_of_Min_Amount'] # Categorico
    payment_behavior = artifact['Payment_Behaviour'] # Categorico
    credit_age = artifact['Credit_History_Age'] # Str

    # Passando a idade para valor numerico em anos
    match = re.search(r'(\d+)\s*Years?\s*and\s*(\d+)\s*Months?', credit_age)
    credit_age_years = float(match.group(1)) + float(match.group(2)) / 12
    artifact['Credit_History_Age'] = credit_age_years
    artifact['Credit_History_Age_Years'] = artifact.pop('Credit_History_Age')

    # Tratando as variaveis categoricas
    credit_mix_dict = {
        'Bad': 0,
        'Good': 1,
        'Standard': 2
    }

    payment_min_dict = {
        'No': 0,
        'Yes': 1
    }

    payment_behavior_dict = {
        'High_spent_Large_value_payments': 1,
        'High_spent_Medium_value_payments': 2,
        'High_spent_Small_value_payments': 3,
        'Low_spent_Large_value_payments': 4,
        'Low_spent_Medium_value_payments': 5,
        'Low_spent_Small_value_payments': 6
    }

    artifact['Credit_Mix'] = credit_mix_dict[credit_mix]
    artifact['Payment_of_Min_Amount'] = payment_min_dict[payment_min]
    artifact['Payment_Behaviour'] = payment_behavior_dict[payment_behavior]

    df = pd.DataFrame([artifact])

    df["Debt_per_Card"] = df["Outstanding_Debt"] / (df["Num_Credit_Card"] + 1)
    df['Interest_x_Delay'] = df['Interest_Rate'] * df['Delay_from_due_date']
    df['Interest_x_Debt'] = df['Interest_Rate'] * df['Outstanding_Debt']
    df["Debt_to_Income"] = df["Outstanding_Debt"] / df['Monthly_Inhand_Salary']
    df["EMI_to_Salary"] = df["Total_EMI_per_month"] / df['Monthly_Inhand_Salary']
    df["Balance_to_Salary"] = df["Monthly_Balance"] / df['Monthly_Inhand_Salary']
    df["Debt_per_Card"] = df["Outstanding_Debt"] / (df["Num_Credit_Card"] + 1)

    return df


def make_pred(
        model,
        input: pd.DataFrame
    ):

    y_pred = model.predict_proba(input)

    pred_dict = {
        0: "Good",
        1: "Poor",
        2: "Standard"
    }

    max_prob_index = list(y_pred[0]).index(max(y_pred[0]))

    return {"prediction":  pred_dict[max_prob_index],
    "class_": max_prob_index,
    "probabilities":{
        'Good': float(y_pred[0][0]),
        'Poor': float(y_pred[0][1]),
        'Standard': float(y_pred[0][2])
    }}