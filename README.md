# Real-time News RAG with Fact-Checking

## Overview
A Retrieval-Augmented Generation (RAG) system for real-time news ingestion, misinformation detection, fact-checking, and source credibility scoring. Built with Streamlit, HuggingFace Sentence Transformers, and ChromaDB.

## Features
- Real-time news ingestion
- Misinformation detection
- Source credibility scoring
- Fact-checking with evidence retrieval
- User-friendly web interface

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app/main.py`

## Project Structure
- `app/` – Streamlit UI
- `ingestion/` – News ingestion scripts
- `retrieval/` – Vector DB and retrieval logic
- `fact_checking/` – Fact-checking modules
- `scoring/` – Source credibility scoring
- `utils/` – Shared utilities

## Deployment
Deployable on HuggingFace Spaces or Streamlit Cloud.
