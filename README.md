# 🧊 The Fridge Survival Guide

AI-powered fridge assistant that detects ingredients from a photo using Gemini Vision 
and recommends recipes using TF-IDF cosine similarity — built on Google Cloud Vertex AI.

## How it works
1. Upload a photo of your fridge
2. Gemini Vision detects all visible ingredients
3. The recommendation engine matches them against a database of 130+ recipes
4. Get ranked recipes with step-by-step instructions

## Tech Stack
- **Vision:** Google Gemini 2.0 Flash via Vertex AI
- **Recommendation:** TF-IDF + Cosine Similarity (scikit-learn)
- **Interface:** Gradio
- **Cloud:** Google Cloud Vertex AI Workbench

## Setup
```bash
git clone https://github.com/catoralonso/The-Fridge-Survival-Guide.git
cd The-Fridge-Survival-Guide
pip install -r requirements.txt

# Configura tu proyecto de Google Cloud
export GCP_PROJECT_ID=$(gcloud config get-value project)
export GCP_LOCATION="us-central1"

python app_gradiov3.py
```
