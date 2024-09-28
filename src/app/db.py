import chromadb
import llama_cpp
from chromadb.api.models.Collection import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from sentence_transformers import SentenceTransformer

from config import device


N_RETRIEVES=10
LLM_GPU_LAYERS=-1


class HandlerLLM:
    def __init__(self, llm_name: str) -> None:
        self.rag_model = llama_cpp.Llama(
            model_path=llm_name,
            split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            n_gpu_layers=LLM_GPU_LAYERS,
            offload_kqv=True,
        )

    def get_response(
        self,
        context: str,
        question: str,
        temperature: float,
        top_p: float
    ) -> str:
        response = self.rag_model.create_chat_completion(
            messages = [
                {
                    "role": "system", "content":
                    """
                    Используя информацию, содержащуюся в контексте, дай полный ответ на вопрос. Отвечай только на поставленный вопрос, ответ должен соответствовать вопросу. Если ответ не может быть выведен из контекста, не отвечай.
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
            ],
            temperature=temperature, 
            top_p=top_p
        )
        return response['choices'][0]['message']['content']


class HandlerRAG:
    def __init__(self, retriever: str, llm: str, reranker: str, collection: Collection) -> None:        
        self.retriever = SentenceTransformer(retriever).to(device)
        self.reranker = reranker
        self.collection = collection
        self.llm = HandlerLLM(llm)
    
    def retrieve(self, question: str):
        retrieved_data = self.collection.query(
            query_embeddings=self.retriever.encode(question).tolist(),
            n_results=N_RETRIEVES,
        )
        return retrieved_data

    def get_context(self, question: str):
        candidates = self.retrieve(question)
        # TODO: make rerank
        return candidates[0]


class ChromaRAG:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.db_settings = Settings(allow_reset=True, anonymized_telemetry=False)
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            ssl=False,
            headers=None,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
    
    def create_collection(self, name: str) -> Collection:
        return self.client.get_or_create_collection(name)
    
    def get_collection(self, name: str) -> Collection:
        return self.client.get_collection(name)

    def remove_collection(self, name: str) -> None:
        return self.client.delete_collection(name)


        
        
        

    def add_to_collection(self, name: )