# db_utils.py

import sys
from typing import List


def _patch_sqlite():
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


_patch_sqlite()


import chromadb
from chromadb.utils import embedding_functions


class ChromaDBClient:
    def __init__(self, collection_name: str, model_name: str, chroma_host: str = None, chroma_port: int = None):
        """
        Initialize ChromaDB client and create/get the collection.
        Optionally connect to an external ChromaDB instance running on a specific host/port.

        :param collection_name: Name of the collection to use
        :param model_name: Name of the embedding model to use
        :param chroma_host: Optional host of the ChromaDB instance (for connecting to Docker)
        :param chroma_port: Optional port of the ChromaDB instance (for connecting to Docker)
        """
        _patch_sqlite()
        self.collection_name = collection_name
        self.model_name = model_name
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.client = None
        self.collection = None
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    def connect(self):
        """
        Connect to the ChromaDB instance. If chroma_host and chroma_port are provided,
        it will connect to the external ChromaDB instance; otherwise, it will connect locally.
        """
        if self.chroma_host and self.chroma_port:
            # Connect to the external ChromaDB instance running on Docker
            self.client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        else:
            raise ValueError('Invalid ChromaDB host or port. Please, check your credentials!')

    def add_documents(self, documents: List[str], embeddings: List[List[float]], ids: List[str]):
        """
        Add documents and their corresponding embeddings to the collection.

        :param documents: List of documents to add
        :param embeddings: Precomputed embeddings for the documents
        :param ids: List of unique IDs for the documents
        """
        if self.collection is None:
            raise Exception("Not connected to ChromaDB. Please call connect() first.")
        self.collection.add(documents=documents, embeddings=embeddings, ids=ids)

    def query(self, query_embedding: List[float], n_results: int = 2):
        """
        Query the collection based on a given embedding.

        :param query_embedding: The embedding vector for the query
        :param n_results: Number of results to return
        :return: List of document results
        """
        if self.collection is None:
            raise Exception("Not connected to ChromaDB. Please call connect() first.")
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results['documents']


def encode_text(model, text: str):
    """
    Helper function to encode text using the model.

    :param model: The embedding model
    :param text: Text to encode
    :return: The embedding for the text
    """
    return model.encode(text).tolist()
