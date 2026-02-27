import json
import vertexai
from vertexai.generative_models import GenerativeModel, Image as VertexImage

# ──────────────────────────────────────────────
#  CONFIGURACIÓN VERTEX AI
# ──────────────────────────────────────────────

PROJECT_ID = "qwiklabs-gcp-00-0fac841e62df"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.0-flash-001")

# ──────────────────────────────────────────────
#  INGREDIENTES A DESCARTAR
# ──────────────────────────────────────────────

DISCARD = {
    "ensalada de frutas", "ensalada de pasta", "mermelada de frutas",
    "aderezo para ensalada", "fiambre", "verduras de hoja", "hierbas",
    "aceite de cocina", "botella", "verdura", "comida",
    "bebida embotellada", "agua",
}

PROMPT = """
Mira esta imagen de una nevera y lista TODOS los ingredientes y productos visibles.
Devuelve ÚNICAMENTE un array JSON, sin ningún otro texto. Formato:
[
    {"name": "nombre del ingrediente", "confidence": 0.95},
    ...
]
Reglas:
- Usa nombres simples en ESPAÑOL (ej: "zanahoria" no "zanahoria fresca orgánica")
- Incluye todo lo visible: frutas, verduras, bebidas, condimentos, lácteos, sobras
- Confidence: 0.9 si se ve claramente, 0.7 si se ve parcialmente, 0.5 si hay incertidumbre
- NO incluyas nombres de marcas, pon el ingrediente real (ej: "zumo de naranja" no "Tropicana")
"""

# ──────────────────────────────────────────────
#  DETECCIÓN CON VERTEX AI
# ──────────────────────────────────────────────

def detect_gemini(image_path: str, api_key: str = None) -> list:
    """
    Envía la imagen a Gemini 1.5 Flash via Vertex AI.
    api_key se mantiene por compatibilidad pero no se usa.
    """
    image = VertexImage.load_from_file(image_path)
    response = model.generate_content(
        [PROMPT, image],
        generation_config={"temperature": 0.1},
    )
    text = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ──────────────────────────────────────────────
#  LIMPIEZA Y NORMALIZACIÓN
# ──────────────────────────────────────────────

def _normalize(name: str) -> str:
    import unicodedata
    n = name.lower().strip()
    n = unicodedata.normalize("NFD", n)
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    if n.endswith("oes"):
        n = n[:-2]
    elif n.endswith("s") and not n.endswith("ss"):
        n = n[:-1]
    return n


def clean_ingredients(detections: list, min_confidence: float = 0.5) -> list:
    seen = set()
    cleaned = []
    for item in sorted(detections, key=lambda x: -x["confidence"]):
        if item["confidence"] < min_confidence:
            continue
        name_norm = _normalize(item["name"])
        if name_norm in DISCARD or item["name"].lower() in DISCARD:
            continue
        if name_norm not in seen:
            seen.add(name_norm)
            cleaned.append({
                "name": item["name"].lower(),
                "confidence": item["confidence"],
            })
    return cleaned


# ──────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────

def detectar_ingredientes(image_path: str, api_key: str = None) -> list:
    raw = detect_gemini(image_path)
    cleaned = clean_ingredients(raw)
    return [item["name"] for item in cleaned]


# ──────────────────────────────────────────────
#  PRUEBA RÁPIDA
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "foto_nevera.png"
    print(f"\n🔍 Analizando: {img}\n")
    raw = detect_gemini(img)
    cleaned = clean_ingredients(raw)
    print(f"INGREDIENTES DETECTADOS ({len(cleaned)}):")
    for item in cleaned:
        print(f"  {item['name']:<30} {item['confidence']:.0%}")
    print(f"\nLista para el recomendador:")
    print([i["name"] for i in cleaned])
