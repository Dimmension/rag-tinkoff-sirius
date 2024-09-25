import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RETRIEVER_MODEL_NAME = 'ai-forever/ru-en-RoSBERTa'
DATASET_NAME = 'kuznetsoffandrey/sberquad'
PATH_TO_EMBEDDINGS = 'data/context_embeddings.npy'
PATH_TO_DATASET = 'data/dataset/'
