# Credit Scoring

## 1. Notebooks de analise dos dados e modelagem

## 2. Docker: 

### 2.1. Build dos Containers e Imagens

```python
  docker compose up --build
```

### 2.2. Build do Container de Treino e Execucao da Imagem

```python
  docker compose up --build train
```

```python
  docker run --rm -v model_store:/app/models credit-score-train
```

### 2.2. Build do Container da API e Execucao da Imagem

```python
  docker compose up --build api
```

```python
  docker run -d -p 8000:8000 --name container-api -v model_store:/app/models credit-score-api
```

## 3. API

Iniciar a API:

```python
  uvicorn src.services.api:app --host 0.0.0.0 --port 8000
```


Exemplo de Input:

```python

{
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

```

