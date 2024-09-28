import os
from config import RETRIEVER_MODEL_NAME, RERANKER_MODEL_NAME, MODEL_CACHE
from sentence_transformers import CrossEncoder, SentenceTransformer


class RAGHandler:

    def __init__(
        self,
        retriever_model_name=RETRIEVER_MODEL_NAME,
        reranker_model_name=RERANKER_MODEL_NAME,
        cache_dir=MODEL_CACHE,
    ) -> None:

        assert retriever_model_name is not None or reranker_model_name is not None or llm_model_name is not None, f"Error type of some model: {retriever_model_name}, {reranker_model_name}, {llm_model_name}, please set all env RETRIEVER_MODEL_NAME, RERANKER_MODEL_NAME, LLM_MODEL_NAME or pass it as an attribute!"

        if cache_dir is None:
            cache_dir = os.path.abspath('./hf_cache')

        self.retriever_model = SentenceTransformer(
            RETRIEVER_MODEL_NAME, 
            cache_folder=cache_dir
        ).to('cuda')

        # self.reranker_model = CrossEncoder(
        #     RERANKER_MODEL_NAME, 
        #     max_length=512,
        #     cache_dir=cache_dir,
        # )

    def retrieve_and_rerank(  
        self,
        query: str,
    ) -> list:


        results = self.reranker_model.rank(
            query,
            self.corpus,
            top_k=10,
        )

        return results
