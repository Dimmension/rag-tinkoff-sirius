import os
import torch

os.environ['CUDA_VISIBLE_DEVICES']='3'

CHROMA_HOST = '172.22.100.166'
CHROMA_PORT = 4810

APP_HOST = '127.0.0.1'
APP_PORT = 4830

CHROMA_COLLECTION_NAME = 'sberquad_rag'
DATASET_NAME = 'kuznetsoffandrey/sberquad'
LLM_PATH = 'models/meta-llama-3.1-8b-instruct.Q6_K.gguf'
RETRIEVER_NAME = 'ai-forever/ru-en-RoSBERTa'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_GPU_LAYERS = -1
PROMPT_RAG = [
    {
        "role": "system",
        "content": """Используя информацию, содержащуюся в контексте, дай полный ответ на вопрос. Отвечай только на поставленный вопрос, ответ должен соответствовать вопросу. Если ответ не может быть выведен из контекста, не отвечай.Оформляй красиво итоговый ответ""",
    },
    {
        "role": "user", 
        "content":
        """
        Контекст: {}
        ---
        Вот вопрос на который тебе надо ответить.
        Вопрос: {}
        """,
    },
]
