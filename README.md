# 🧊 The Fridge Survival Guide

AI-powered fridge assistant that detects ingredients from a photo using Gemini Vision 
and recommends recipes using TF-IDF cosine similarity — built on Google Cloud Vertex AI.

## How it works
1. Upload a photo of your fridge
2. Gemini Vision detects all visible ingredients
3. The recommendation engine matches them against a database of 130+ recipes
4. Get ranked recipes with step-by-step instructions

## Tech Stack
- **Vision:** Google Gemini 1.5 Flash via Vertex AI
- **Recommendation:** TF-IDF + Cosine Similarity (scikit-learn)
- **Interface:** Gradio
- **Cloud:** Google Cloud Vertex AI Workbench

## Setup
```bash
git clone https://github.com/vuestro-usuario/fridge-survival-guide.git
cd fridge-survival-guide
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"
python app_gradio.py
```

## Project Structure
```
app_gradio.py        → Gradio interface
vision.py            → Gemini Vision module
recommender.py       → TF-IDF recommendation engine
recetas_backend_proceso_ultra.json  → Recipes database
```
