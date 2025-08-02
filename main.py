from fastapi import FastAPI
from pydantic import BaseModel
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Update this with your real PDF file
data_path = "BAJHLIP23020V012223.pdf"

class QueryRequest(BaseModel):
    query: str

# PDF reading
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# Load embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
document_text = extract_text_from_pdf(data_path)
sentences = document_text.split(".")
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

# Clause matching
def retrieve_relevant_clauses(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    top_results = scores.topk(3)
    return [sentences[i] for i in top_results.indices]

@app.post("/api/v1/hackrx/run")
async def run(payload: QueryRequest):
    query = payload.query
    relevant_clauses = retrieve_relevant_clauses(query)
    decision = "Approved" if "covered" in str(relevant_clauses).lower() else "Rejected"

    return {
        "decision": decision,
        "amount": 50000 if decision == "Approved" else None,
        "justification": " ".join(relevant_clauses),
        "clauses_referenced": [
            {
                "document": data_path,
                "clause_excerpt": clause.strip()
            } for clause in relevant_clauses
        ]
    }
