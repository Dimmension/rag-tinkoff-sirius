import chromadb
import llama_cpp
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import llama_cpp
from config import device

class DatasetRAG:
    def __init__(self, dataset: str) -> None:
        self.dataset = load_dataset(dataset)
    
    def save_to_disk(self, save_path: str) -> None:
        self.dataset.save_to_disk(save_path)


class HandlerLLM:
    def __init__(self, model_name: str) -> None:
        self.rag_model = llama_cpp.Llama(
            model_path=model_name,
            split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            n_gpu_layers=-1,
            offload_kqv=True,
        )

    def get_response(self, context: str, question: str) -> str:
        response = self.rag_model.create_chat_completion(messages = [
            {
                "role": "system", "content":
        """
        Используя информацию, содержащуюся в контексте, дай полный ответ на вопрос. Отвечай только на поставленный вопрос, ответ должен соответствовать вопросу. Если ответ не может быть выведен из контекста, не отвечай.
        Оформляй красиво итоговый ответ
        """,
            },
            {
                "role": "user", "content":
        f"""
        Контекст: {context}
        ---
        Вот вопрос на который тебе надо ответить.
        Вопрос: {question}
        """,
            },
        ], temperature=0.2, top_p=0.80)
        return response['choices'][0]['message']['content']


class HandlerRAG:
    def __init__(self, retriever: str, reranker: str) -> None:        
        self.retriever = SentenceTransformer(retriever).to(device)
        self.reranker = reranker
    
    def retrieve(self, question: str, collection: Collection):
        retrieved_data = collection.query(
            query_embeddings=self.retriever.encode(question).tolist(),
            n_results=10,
        )
        return retrieved_data['documents']

    def get_context(self, question: str, collection: Collection):
        candidates = self.retrieve(question, collection)
        return candidates[0][0]


class ChromaRAG:
    def __init__(self, host: str, port: int, dataset: str, retriever: SentenceTransformer) -> None:
        self.host = host
        self.port = port
        self.db_settings = Settings(allow_reset=True, anonymized_telemetry=False)
        self.dataset = load_dataset(dataset)
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            ssl=False,
            headers=None,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        self.retriever = retriever
        self.collection = None
        self.emb_func = None

    def add_emb_function(self, model: str):
        self.emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
        return self.emb_func
    
    def get_collection(self, name: str) -> Collection:
        self.collection = self.client.get_or_create_collection(name, embedding_function=self.emb_func)
        return self.collection
    
    def remove_collection(self, name: str) -> None:
        self.collection = None
        return self.client.delete_collection(name)

    def get_embeddings(self, model, save_path: str = '', split: str = 'train', column: str = 'context') -> tuple[list, list]:
        context_sentences = list(set(self.dataset[split][column]))
        context_embeddings = None
        try:
            context_embeddings = np.load(save_path)
        except Exception:
            context_embeddings = model.encode(context_sentences)
            np.save(save_path, context_embeddings)
        return (context_sentences, context_embeddings)

    def add_embeddings_to_collection(self, context_sentences, context_embeddings) -> None:
        self.collection.add(
            documents=context_sentences,
            embeddings=context_embeddings.tolist(),
            ids=[str(i) for i in range(len(context_sentences))]
        )
    
    def setup_collection(self, collection_name: str):
        self.add_emb_function('ai-forever/ru-en-RoSBERTa')
        self.get_collection(collection_name)
        self.add_embeddings_to_collection(*self.get_embeddings(self.retriever, 'data/context_embeddings.npy'))
        return self.collection

