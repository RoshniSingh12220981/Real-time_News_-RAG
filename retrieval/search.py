from sentence_transformers import SentenceTransformer
import chromadb
from fact_checking.check import fact_check
from scoring.credibility import get_source_credibility

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("news")

def search_news(query, top_k=3):
    query_emb = model.encode(query)
    results = collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k)
    output = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        credibility = get_source_credibility(meta['source'])
        fact, evidence = fact_check(query, doc)
        output.append({
            'source': meta['source'],
            'credibility': credibility,
            'fact_check': fact,
            'evidence': evidence,
            'context': doc,
            'misinfo_verdict': meta.get('misinfo_verdict', 'Unknown'),
            'misinfo_explanation': meta.get('misinfo_explanation', '')
        })
    return output
