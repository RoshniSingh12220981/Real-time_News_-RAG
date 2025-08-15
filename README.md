# Real-time News RAG with Fact-Checking

## Overview
A Retrieval-Augmented Generation (RAG) system for real-time news ingestion, misinformation detection, fact-checking, and source credibility scoring. Built with Streamlit, HuggingFace Sentence Transformers, and ChromaDB.

## Features
- Real-time news ingestion
- Misinformation detection
- Source credibility scoring
- Fact-checking with evidence retrieval
- User-friendly web interface

##ScreenShot of Working Demo 
<img width="1366" height="683" alt="Screenshot (892)" src="https://github.com/user-attachments/assets/0e224ffd-5fbd-46d2-a2d5-5209168f0463" />
<img width="1366" height="666" alt="Screenshot (894)" src="https://github.com/user-attachments/assets/9d5d8254-8ddd-4f22-b652-69feeeb560ab" />
<img width="1366" height="602" alt="image" src="https://github.com/user-attachments/assets/4a18baae-7f9b-4084-a4ad-0c0726b820af" />
<img width="1366" height="673" alt="Screenshot (897)" src="https://github.com/user-attachments/assets/39f9eb4e-3b9d-4109-a1c7-95d67b0dabc4" />





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
