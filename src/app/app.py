import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER
from utils import ChromaRAG, HandlerLLM, HandlerRAG
import config

# Pre-load the Llama and SentenceTransformer models
llama_model = Llama(
    model_path=config.LLM_PATH,
    split_mode=LLAMA_SPLIT_MODE_LAYER,
    n_gpu_layers=config.N_GPU_LAYERS,
    offload_kqv=True
)

# Load the SentenceTransformer retriever model
retriever_model = SentenceTransformer(config.RETRIEVER_NAME).to(config.DEVICE)

# Instantiate classes with pre-loaded models
llm_handler = HandlerLLM(llama_model)
rag_handler = HandlerRAG(retriever_model)
chroma_rag = ChromaRAG(
    host=config.CHROMA_HOST,
    port=config.CHROMA_PORT,
    dataset_name=config.DATASET_NAME,
    retriever=retriever_model
)

# Setup Chroma collection
chroma_rag.setup_collection(config.CHROMA_COLLECTION_NAME, config.RETRIEVER_NAME)

# Define API request model
class Query(BaseModel):
    question: str

# Create FastAPI app instance
app = FastAPI()

@app.post("/query", response_class=JSONResponse)
def query_rag(query_data: Query):
    question = query_data.question
    try:
        # Retrieve context using RAG
        retrieved_context = rag_handler.get_context(
            question, chroma_rag.get_collection(config.CHROMA_COLLECTION_NAME)
        )
        # Generate LLM response using the retrieved context
        llm_response = llm_handler.get_response(context=retrieved_context, question=question)

        return {"answer": llm_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Entry point for running the API with Uvicorn
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        log_level="info",
    )