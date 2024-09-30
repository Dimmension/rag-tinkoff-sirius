import os
import torch

# Set the specific GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Configuration for Chroma and application hosts and ports
CHROMA_HOST = '172.22.100.166' # 127.0.0.1
APP_HOST = '127.0.0.1'
CHROMA_PORT = 4810
APP_PORT = 4830

# Hugging Face model link
LLM_LINK = 'https://huggingface.co/SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/meta-llama-3.1-8b-instruct.Q6_K.gguf'

# Chroma collection and dataset configurations
CHROMA_COLLECTION_NAME = 'sberquad_rag'
DATASET_NAME = 'kuznetsoffandrey/sberquad'

# Model and retriever configurations
LLM_NAME = 'meta-llama-3.1-8b-instruct.Q6_K.gguf'
LLM_PATH = 'models/meta-llama-3.1-8b-instruct.Q6_K.gguf'
RETRIEVER_NAME = 'ai-forever/ru-en-RoSBERTa'

# Set device based on GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration for number of layers to allocate on GPU (use -1 for full offloading)
N_GPU_LAYERS = -1

# Prompt template for RAG (Retrieval-Augmented Generation)
PROMPT_RAG = [
    {
        "role": "system",
        "content": """Используя информацию, содержащуюся в контексте, дай полный ответ на вопрос. Отвечай только на поставленный вопрос, ответ должен соответствовать вопросу. Если ответ не может быть выведен из контекста, не отвечай. Оформляй красиво итоговый ответ."""
    },
    {
        "role": "user",
        "content": """
        Контекст: {}
        ---
        Вот вопрос на который тебе надо ответить.
        Вопрос: {}
        """
    },
]