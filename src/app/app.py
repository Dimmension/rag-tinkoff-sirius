import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from db import (ChromaRAG, HandlerLLM, HandlerRAG)
from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER
from config import DEVICE, N_GPU_LAYERS
from llama_cpp import Llama 
from sentence_transformers import SentenceTransformer 
 
# Pre-load the Llama and SentenceTransformer models 
llama_model = Llama(model_path="models/meta-llama-3.1-8b-instruct.Q6_K.gguf", split_mode=LLAMA_SPLIT_MODE_LAYER, n_gpu_layers=N_GPU_LAYERS, offload_kqv=True) 
retriever_model = SentenceTransformer('ai-forever/ru-en-RoSBERTa').to(DEVICE) 
 
# Instantiate classes with pre-loaded models 
llm_handler = HandlerLLM(llama_model) 
rag_handler = HandlerRAG(retriever_model) 
chroma_rag = ChromaRAG(host="172.22.100.166", port=4810, dataset_name="kuznetsoffandrey/sberquad", retriever=retriever_model)
chroma_rag.setup_collection('sberquad_rag', 'ai-forever/ru-en-RoSBERTa')

# Define your API request model
class Query(BaseModel):
    question: str

# Create the FastAPI app
app = FastAPI()

@app.post("/query")
def query_rag(query_data: Query):
    question = query_data.question
    try:
        retrieved_context = rag_handler.get_context(question, chroma_rag.get_collection('sberquad_rag'))  
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
