import chromadb
import numpy as np
import logging
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from llama_cpp import Llama
from config import PROMPT_RAG
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DatasetRAG:
    """
    A class to load and save datasets for retrieval-augmented generation (RAG) tasks.
    
    Attributes:
        dataset (Dataset): The loaded dataset.
    """
    def __init__(self, dataset_name: str):
        """
        Initialize DatasetRAG with a specified dataset.
        
        Args:
            dataset_name (str): The name of the dataset to load.
        """
        try:
            self.dataset = load_dataset(dataset_name)
            logging.info(f"Dataset {dataset_name} loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            raise e

    def save_to_disk(self, save_path: str) -> None:
        """
        Save the dataset to disk at the specified path.
        
        Args:
            save_path (str): The file path to save the dataset.
        """
        try:
            self.dataset.save_to_disk(save_path)
            logging.info(f"Dataset saved to {save_path}.")
        except Exception as e:
            logging.error(f"Failed to save dataset to {save_path}: {e}")
            raise e


class HandlerLLM:
    """
    A class to handle interaction with the Llama language model.

    Attributes:
        rag_model (Llama): The Llama model instance for generating responses.
    """
    def __init__(self, llama_model: Llama):
        """
        Initialize the Llama model handler.
        
        Args:
            llama_model (Llama): Pre-loaded Llama model instance.
        """
        self.rag_model = llama_model

    def get_response(self, context: str, question: str, temp: float = 0.2, top_p: float = 0.8) -> str:
        """
        Generate a response from the Llama model based on the provided context and question.
        
        Args:
            context (str): The context to consider.
            question (str): The question to answer.
            temp (float): Temperature for sampling. Defaults to 0.2.
            top_p (float): Top-p for nucleus sampling. Defaults to 0.8.
        
        Returns:
            str: The generated response.
        """
        try:
            prompt = [
                PROMPT_RAG[0],
                {'role': 'user', 'content': PROMPT_RAG[1]['content'].format(context, question)},
            ]
            response = self.rag_model.create_chat_completion(prompt, temperature=temp, top_p=top_p)
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Failed to generate response from Llama model: {e}")
            raise e


class HandlerRAG:
    """
    A class to handle retrieval and reranking for Retrieval-Augmented Generation (RAG).

    Attributes:
        retriever (SentenceTransformer): A sentence transformer for question embedding.

    """
    def __init__(self, retriever: SentenceTransformer):
        """
        Initialize the HandlerRAG with pre-loaded retriever and reranker models.
        
        Args:
            retriever (SentenceTransformer): Pre-loaded retriever model.
        """
        self.retriever = retriever
    
    def retrieve(self, question: str, collection: Collection) -> list:
        """
        Retrieve relevant documents based on the question.
        
        Args:
            question (str): The question for document retrieval.
            collection (Collection): The ChromaDB collection to query.
        
        Returns:
            list: The retrieved documents.
        """
        try:
            embeddings = self.retriever.encode(question, convert_to_tensor=True).cpu().tolist()
            results = collection.query(query_embeddings=embeddings, n_results=10)
            return results['documents']
        except Exception as e:
            logging.error(f"Failed to retrieve documents: {e}")
            raise e

    def get_context(self, question: str, collection: Collection) -> str:
        """
        Retrieve the best matching document based on the question.
        
        Args:
            question (str): The question for which to retrieve context.
            collection (Collection): The ChromaDB collection to query.
        
        Returns:
            str: The most relevant document as context.
        """
        try:
            candidates = self.retrieve(question, collection)
            return candidates[0][0] if candidates else None
        except Exception as e:
            logging.error(f"Failed to get context for question: {e}")
            raise e


class ChromaRAG:
    """
    A class for integrating ChromaDB with RAG to handle document retrieval and embedding functions.
    
    Attributes:
        client (HttpClient): The ChromaDB client.
        dataset (Dataset): The loaded dataset for embeddings.
        retriever (SentenceTransformer): A sentence transformer model for embedding generation.
        emb_func (EmbeddingFunction): Embedding function for the collection.
        collection (Collection): The current ChromaDB collection.
    """
    def __init__(self, host: str, port: int, dataset_name: str, retriever: SentenceTransformer):
        """
        Initialize ChromaRAG with ChromaDB client, dataset, and pre-loaded retriever model.
        
        Args:
            host (str): The host for the ChromaDB client.
            port (int): The port for the ChromaDB client.
            dataset_name (str): The name of the dataset.
            retriever (SentenceTransformer): Pre-loaded retriever model.
        """
        try:
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=False,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            self.dataset = load_dataset(dataset_name)
            self.retriever = retriever
            self.collection = None
            logging.info(f"ChromaDB client initialized on {host}:{port} with dataset {dataset_name}.")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaRAG: {e}")
            raise e
    
    def add_emb_function(self, model_name: str):
        """
        Add an embedding function using the specified model.
        
        Args:
            model_name (str): The name of the model to use for the embedding function.
        """
        try:
            self.emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
            logging.info(f"Embedding function added for model {model_name}.")
        except Exception as e:
            logging.error(f"Failed to add embedding function: {e}")
            raise e

    def get_collection(self, name: str) -> Collection:
        """
        Get or create a ChromaDB collection.
        
        Args:
            name (str): The name of the collection.
        
        Returns:
            Collection: The ChromaDB collection.
        """
        try:
            if not self.collection:
                self.collection = self.client.get_or_create_collection(name, embedding_function=self.emb_func)
            return self.collection
        except Exception as e:
            logging.error(f"Failed to get or create collection {name}: {e}")
            raise e

    def remove_collection(self, name: str) -> None:
        """
        Remove the specified collection from ChromaDB.
        
        Args:
            name (str): The name of the collection to remove.
        """
        try:
            if self.collection:
                self.collection = None
            self.client.delete_collection(name)
            logging.info(f"Collection {name} removed.")
        except Exception as e:
            logging.error(f"Failed to remove collection {name}: {e}")
            raise e

    def get_embeddings(self, save_path: str = 'data/context_embeddings.npy', split: str = 'train', column: str = 'context') -> tuple[list, np.ndarray]:
        """
        Get embeddings for the dataset using the pre-loaded retriever model.
        
        Args:
            save_path (str): Path to save or load embeddings. Defaults to ''.
            split (str): Dataset split to use. Defaults to 'train'.
            column (str): Column in the dataset to embed. Defaults to 'context'.
        
        Returns:
            tuple: A tuple of (context sentences, embeddings).
        """
        context_sentences = list(set(self.dataset[split][column]))
        try:
            if os.path.exists(save_path):
                context_embeddings = np.load(save_path)
                logging.info(f"Loaded embeddings from {save_path}.")
            else:
                context_embeddings = self.retriever.encode(context_sentences, batch_size=64, convert_to_tensor=True).cpu().numpy()
                np.save(save_path, context_embeddings)
                logging.info(f"Embeddings generated and saved to {save_path}.")
            return context_sentences, context_embeddings
        except Exception as e:
            logging.error(f"Failed to get embeddings: {e}")
            raise e

    def add_embeddings_to_collection(self, context_sentences: list, context_embeddings: np.ndarray) -> None:
        """
        Add embeddings to the ChromaDB collection.
        
        Args:
            context_sentences (list): List of sentences.
            context_embeddings (np.ndarray): The corresponding embeddings.
        """
        try:
            self.collection.add(
                documents=context_sentences,
                embeddings=context_embeddings.tolist(),
                ids=[str(i) for i in range(len(context_sentences))]
            )
            logging.info("Embeddings added to the collection.")
        except Exception as e:
            logging.error(f"Failed to add embeddings to collection: {e}")
            raise e

    def setup_collection(self, collection_name: str, model_name: str) -> Collection:
        """
        Setup a collection by adding embeddings and setting the embedding function.
        
        Args:
            model_name: (str): The name of model with embedding function
            collection_name (str): The name of the collection.
        
        Returns:
            Collection: The setup ChromaDB collection.
        """
        try:
            self.add_emb_function(model_name)
            self.get_collection(collection_name)
            context_sentences, context_embeddings = self.get_embeddings()
            self.add_embeddings_to_collection(context_sentences, context_embeddings)
            logging.info(f"Collection {collection_name} setup complete.")
            return self.collection
        except Exception as e:
            logging.error(f"Failed to set up collection {collection_name}: {e}")
            raise e
