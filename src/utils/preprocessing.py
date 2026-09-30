import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

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

TARGET_VARIABLE = 'Credit_Score'

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(['ID', 'SSN', 'Name', 'Customer_ID'], axis=1)

    # Extrai o padrao do Credit_History_Age e cria coluna Credit_History_Age_Years numerica
    extracted = df['Credit_History_Age'].str.extract(r'(\d+)\s*Years?\s*and\s*(\d+)\s*Months?')
    df['Credit_History_Age_Years'] = (
        extracted[0].astype(float) + extracted[1].astype(float) / 12
    )
    df = df.drop(columns=['Credit_History_Age'])

    # Trata variaveis com valores esquisitos - '_', 'NM'
    moda_credit_mix = df["Credit_Mix"].mode()[0]
    moda_payment = df["Payment_of_Min_Amount"].mode()[0]

    df.loc[(df['Credit_Mix'] == '_'), 'Credit_Mix'] = moda_credit_mix
    df.loc[(df['Payment_of_Min_Amount'] == 'NM'), 'Payment_of_Min_Amount'] = moda_payment


    NUMERICAL_COLUMNS = ['Age',
        'Annual_Income',
        'Monthly_Inhand_Salary',
        'Num_Bank_Accounts',
        'Num_Credit_Card',
        'Interest_Rate',
        'Num_of_Loan',
        'Delay_from_due_date',
        'Num_of_Delayed_Payment',
        'Changed_Credit_Limit',
        'Num_Credit_Inquiries',
        'Outstanding_Debt',
        'Credit_Utilization_Ratio',
        'Total_EMI_per_month',
        'Amount_invested_monthly',
        'Monthly_Balance',
        'Credit_History_Age_Years'
    ]

    # Limpa colunas numericas
    for col in NUMERICAL_COLUMNS:
        df[col] = df[col].astype(str).str.rstrip('_')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Trata colunas com valores inexistentes
    moda_changed_credit = df["Changed_Credit_Limit"].mode()[0]
    moda_inquiries = df["Num_Credit_Inquiries"].mode()[0]
    moda_loan_type = df["Type_of_Loan"].mode()[0]
    moda_delayed = df["Num_of_Delayed_Payment"].mode()[0]

    media_salary = df["Monthly_Inhand_Salary"].mean()
    media_invested = df["Amount_invested_monthly"].mean()
    media_balance = df["Monthly_Balance"].mean()
    media_history = df["Credit_History_Age_Years"].mean()

    df.loc[df["Credit_History_Age_Years"].isna(), 'Credit_History_Age_Years'] = media_history
    df.loc[df["Num_Credit_Inquiries"].isna(), 'Num_Credit_Inquiries'] = moda_inquiries
    df.loc[df["Num_of_Delayed_Payment"].isna(), 'Num_of_Delayed_Payment'] = moda_delayed
    df.loc[df["Type_of_Loan"].isna(), 'Type_of_Loan'] = moda_loan_type
    df.loc[df["Changed_Credit_Limit"].isna(), 'Changed_Credit_Limit'] = moda_changed_credit

    df.loc[df["Monthly_Inhand_Salary"].isna(), 'Monthly_Inhand_Salary'] = media_salary
    df.loc[df["Amount_invested_monthly"].isna(), 'Amount_invested_monthly'] = media_invested
    df.loc[df["Monthly_Balance"].isna(), 'Monthly_Balance'] = media_balance

    # Remove outliers
    CUTOFF_VARIABLES = [
        'Monthly_Inhand_Salary', 
        'Num_Bank_Accounts', 
        'Num_Credit_Card', 
        'Interest_Rate', 
        'Delay_from_due_date', 
        'Num_Credit_Inquiries', 
        'Total_EMI_per_month'
    ]

    cutoff_values = df[CUTOFF_VARIABLES].quantile(0.985)
    for column in CUTOFF_VARIABLES:
        cutoff = cutoff_values[column]
        df = df.loc[df[column] <= cutoff]

    # Seleciona as colunas desejadas
    GOOD_VARIABLES = [
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
        'Credit_Score'
    ]
    

    # Faz o encoding das variaveis categoricas
    colunas_texto = df.select_dtypes(include='string').columns.tolist()

    oe = OrdinalEncoder()
    df[colunas_texto] = oe.fit_transform(df[colunas_texto])

    # Remove valores negativos
    df = df.where(df >= 0, np.nan)

    # Adiciona features
    df["Debt_per_Card"] = df["Outstanding_Debt"] / (df["Num_Credit_Card"] + 1)
    df['Interest_x_Delay'] = df['Interest_Rate'] * df['Delay_from_due_date']
    df['Interest_x_Debt'] = df['Interest_Rate'] * df['Outstanding_Debt']
    df["Debt_to_Income"] = df["Outstanding_Debt"] / df['Monthly_Inhand_Salary']
    df["EMI_to_Salary"] = df["Total_EMI_per_month"] / df['Monthly_Inhand_Salary']
    df["Balance_to_Salary"] = df["Monthly_Balance"] / df['Monthly_Inhand_Salary']

    return df

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state = 4
) -> tuple:
    X, y = df[FEATURE_VARIABLES], df[TARGET_VARIABLE]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test