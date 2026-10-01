import json
from google import genai
from google.genai import types

client = genai.Client()

SYSTEM_PROMPT = f"""Você explica predições de um modelo de scoring de credito para analistas de negócio. Voce recebera
    valores SHAP com as features mais importantes para o resultado da predicao. Voce deve criar um texto de dois paragrafos analisando os valores SHAP para dar
    uma explicacao profissional do motivo pelo qual o input recebeu o scoring informado. Note que 0 = Good, 1 = Poor, 2 = Standard. Note tambem que a entrada i de 
    top_features corresponde ao valor SHAP i de top_valores.
"""

def llm_explain(payload: dict, classe: int) -> str:
    context = {
        "classe": classe,
        "top_features": [str(f) for f in payload["top_features"]],
        "top_valores": [float(v) for v in payload["top_valores"]]
    }
    resposta = client.models.generate_content(
        model="gemini-3.5-flash",
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
