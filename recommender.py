import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
#  CARGA DE DATOS
# ──────────────────────────────────────────────

def cargar_recetas(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
#  NORMALIZACIÓN
# ──────────────────────────────────────────────

def normalizar(texto: str) -> str:
    import unicodedata
    n = texto.lower().strip()
    n = unicodedata.normalize("NFD", n)
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    return n


# ──────────────────────────────────────────────
#  PREPARAR CORPUS PARA TF-IDF
# ──────────────────────────────────────────────

def preparar_corpus(recetas: list) -> list:
    textos = []
    for r in recetas:
        items_clave = [normalizar(i["item"]) for i in r["ingredientes_clave"]]
        nombre = normalizar(r["nombre"])
        tags = " ".join(r.get("tags", [])).lower()
        base = " ".join(r.get("ingredientes_base", [])).lower()
        contenido = f"{nombre} {' '.join(items_clave)} {tags} {base}"
        textos.append(contenido)
    return textos


# ──────────────────────────────────────────────
#  INICIALIZACIÓN GLOBAL (se llama una vez al arrancar)
# ──────────────────────────────────────────────

def init_vectorizer(recetas: list):
    textos = preparar_corpus(recetas)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textos)
    return vectorizer, tfidf_matrix


# ──────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL DE RECOMENDACIÓN
# ──────────────────────────────────────────────

def recomendar(mis_ingredientes: list, recetas: list, vectorizer, tfidf_matrix, n: int = 5) -> list:
    """
    Devuelve las n mejores recetas para los ingredientes dados.
    Scoring por prioridad:
      1. n_coincidencias:  nº absoluto de ingredientes clave que el usuario tiene
      2. porcentaje_match: % de los ingredientes clave de la receta que se cubren
      3. score_tfidf:      similitud coseno TF-IDF como desempate final
    """
    mis_ingredientes = [normalizar(i) for i in mis_ingredientes]
    set_usuario = set(mis_ingredientes)

    resultados = []

    for idx, receta in enumerate(recetas):
        claves = [normalizar(i["item"]) for i in receta["ingredientes_clave"]]
        set_claves = set(claves)

        coincidencias = set_usuario & set_claves
        n_coinciden = len(coincidencias)

        if n_coinciden == 0:
            continue

        porcentaje_match = n_coinciden / len(set_claves)

        resultados.append({
            "receta": receta,
            "n_coincidencias": n_coinciden,
            "porcentaje_match": porcentaje_match,
            "coincidencias": list(coincidencias),
            "idx": idx,
        })

    if not resultados:
        return []

    # Score TF-IDF como desempate final
    usuario_input = " ".join(mis_ingredientes)
    usuario_vector = vectorizer.transform([usuario_input])
    similitudes = cosine_similarity(usuario_vector, tfidf_matrix).flatten()

    for r in resultados:
        r["score_tfidf"] = float(similitudes[r["idx"]])

    # Orden: 1º más coincidencias absolutas, 2º mayor % receta cubierta, 3º TF-IDF
    resultados = sorted(
        resultados,
        key=lambda x: (x["n_coincidencias"], x["porcentaje_match"], x["score_tfidf"]),
        reverse=True,
    )

    return resultados[:n]


# ──────────────────────────────────────────────
#  PRUEBA RÁPIDA (ejecutar directamente)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ruta = sys.argv[1] if len(sys.argv) > 1 else "recetas_backend_proceso_ultra.json"
    recetas = cargar_recetas(ruta)
    vectorizer, tfidf_matrix = init_vectorizer(recetas)

    ingredientes_test = ["huevo", "patata", "queso"]
    print(f"\n🔍 Ingredientes: {ingredientes_test}\n")

    recs = recomendar(ingredientes_test, recetas, vectorizer, tfidf_matrix, n=5)

    for r in recs:
        receta = r["receta"]
        print(f"🍽️  {receta['nombre']}")
        print(f"   Match: {r['porcentaje_match']*100:.0f}%  |  TF-IDF: {r['score_tfidf']:.3f}")
        print(f"   Coincide con: {r['coincidencias']}")
        print(f"   Dificultad: {receta.get('dificultad')}  |  Tiempo: {receta.get('tiempo_min')} min")
        print()
