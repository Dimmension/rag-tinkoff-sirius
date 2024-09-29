import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chromadb.api.models.Collection import Collection  # Import Collection type
from db import (ChromaRAG, HandlerLLM, HandlerRAG)

# Load your RAG components
rag_handler = HandlerRAG("ai-forever/ru-en-RoSBERTa", None) 
llm_handler = HandlerLLM(model_name="models/meta-llama-3.1-8b-instruct.Q6_K.gguf") 

chroma_client = ChromaRAG("172.22.100.166", 4810, 'kuznetsoffandrey/sberquad', rag_handler.retriever)
chroma_client.setup_collection(collection_name="sberquad_rag")
collection = chroma_client.collection


# Define your API request model
class Query(BaseModel):
    question: str

# Create the FastAPI app
app = FastAPI()

@app.post("/query")
def query_rag(query_data: Query):
    question = query_data.question
    try:
        retrieved_context = rag_handler.get_context(question, collection)  
        llm_response = llm_handler.get_response(context=retrieved_context, question=question)

        return {"answer": llm_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run( 
        "app:app", 
        host="127.0.0.1", 
        port=4830, 
        log_level="info", 
    )

