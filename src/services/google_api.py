import json
from google import genai
from google.genai import types

client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            initial_delay=2.0,
            attempts=5,
            max_delay=30.0,
            exp_base=2.0,
            jitter=1.0,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        ),
        timeout=120 * 1000,
    ),
)

SYSTEM_PROMPT = f"""Você explica predições de um modelo de scoring de credito para analistas de negócio. Voce recebera
    valores SHAP com as features mais importantes para o resultado da predicao. Voce deve criar um texto de dois paragrafos analisando os valores SHAP para dar
    uma explicacao profissional do motivo pelo qual o input recebeu o scoring informado. Note que 0 = Good, 1 = Poor, 2 = Standard. Note tambem que a entrada i de 
    top_features corresponde ao valor SHAP i de top_valores.
"""

def llm_explain(payload: dict, classe: int, model) -> str:
    context = {
        "classe": classe,
        "top_features": [str(f) for f in payload["top_features"]],
        "top_valores": [float(v) for v in payload["top_valores"]]
    }
    resposta = client.models.generate_content(
        model=model,
        contents=f"Explique esta predição:\n\n{json.dumps(context, ensure_ascii=False, indent=2)}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=1500,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    print(resposta)
    return resposta.text
