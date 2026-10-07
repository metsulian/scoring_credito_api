# Classificação de Score de Crédito com Machine Learning

Projeto de ponta a ponta para classificar clientes de um banco em faixas de score de crédito (**Good**, **Standard** ou **Poor**): da análise exploratória ao deploy de uma API com explicações geradas por LLM.

---

## Arquitetura

```
┌──────────────────┐        ┌────────────────────────────┐
│  Serviço: train  │        │       Serviço: api         │
│                  │        │                            │
│ treina o modelo  │        │ FastAPI + Uvicorn          │
│ salva artefatos  │        │ carrega modelo e explainer │
└────────┬─────────┘        │ SHAP + Gemini              │
         │                  └──────────────▲─────────────┘
         │   volume compartilhado          │
         └────────► model_store ───────────┘
                    (/app/artifacts)
```

O Docker Compose sobe primeiro o serviço de treino e só inicia a API depois que o treino termina com sucesso. O modelo e o explainer são compartilhados entre os dois por um volume.

## Estrutura do repositório

```
projeto_fraude/
├── artifacts/              # modelo e explainer treinados (.joblib)
├── data/                   # dados (não versionados)
├── notebooks/
│   ├── eda.ipynb           # limpeza, EDA e feature engineering
│   └── modeling.ipynb      # treino e comparação de modelos
├── requirements/
│   ├── api.txt             # dependências da API
│   └── train.txt           # dependências do treino
├── src/
│   ├── functions/
│   │   ├── database.py
│   │   ├── model.py        # predição
│   │   └── shap.py         # explicabilidade
│   ├── services/
│   │   ├── api.py          # endpoints FastAPI
│   │   └── google_api.py   # integração com Gemini
│   ├── utils/
│   │   ├── jobs.py         # salvar/carregar artefatos
│   │   └── preprocessing.py
│   ├── config.py
│   └── train.py            # script de treino
├── Dockerfile              # multi-stage: train e api
├── docker-compose.yml
└── README.md
```

## Como executar

### Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) e Docker Compose
- Uma chave de API do [Google AI Studio](https://aistudio.google.com/) (Gemini)

### 1. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
GEMINI_API_KEY=sua_chave_aqui
```

### 2. Suba os serviços

```bash
docker compose up --build
```

O serviço `train` treina o modelo e salva os artefatos no volume; em seguida a API fica disponível em `http://localhost:8000`.

A documentação interativa (Swagger) está em `http://localhost:8000/docs`.

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Verifica se a API está no ar |
| `POST` | `/predict` | Retorna a classe prevista e as probabilidades |
| `POST` | `/explain` | Gera a explicação em linguagem natural para uma classe informada |
| `POST` | `/predict-explain` | Faz a predição e já devolve a explicação |

### Exemplo de requisição

```bash
curl -X POST "http://localhost:8000/predict-explain" \
  -H "Content-Type: application/json" \
  -d '{
    "Total_EMI_per_month": 120.5,
    "Monthly_Inhand_Salary": 3500,
    "Monthly_Balance": 400,
    "Credit_Mix": "Standard",
    "Payment_of_Min_Amount": "Yes",
    "Payment_Behaviour": "Low_spent_Small_value_payments",
    "Num_Bank_Accounts": 4,
    "Num_Credit_Card": 5,
    "Interest_Rate": 14,
    "Delay_from_due_date": 15,
    "Num_Credit_Inquiries": 6,
    "Outstanding_Debt": 1200,
    "Credit_History_Age": "10 Years and 3 Months"
  }'
```

### Exemplo de resposta

```json
{
  "prediction": "Standard",
  "class_": 2,
  "probabilities": { "Good": 0.08, "Poor": 0.21, "Standard": 0.71 },
  "llm_response": "O cliente foi classificado como Standard principalmente por..."
}
```