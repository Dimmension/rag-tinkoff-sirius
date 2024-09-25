import argparse
import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from config.logger import setup_logger
from config.settings import device

# Argument parser configuration
parser = argparse.ArgumentParser()

parser.add_argument('--model', dest='model', default=None, help="HuggingFace model name for embedding creation")
parser.add_argument('--dataset', dest='dataset', default=None, help="HuggingFace dataset name to load")
parser.add_argument('--emb-save-path', dest='emb_save_path', default=None, help="File path to save the embeddings")
parser.add_argument('--ds-save-path', dest='ds_save_path', default=None, help="File path to save the dataset")


args = parser.parse_args()
logger = setup_logger('create_embeddings.log')


def load_and_save_dataset(hf_ds_name: str, ds_save_path: str) -> Dataset:
    """
    Loads a dataset from HuggingFace by name and saves it to disk.

    Args:
        hf_ds_name (str): The HuggingFace dataset name to load.
        ds_save_path (str): The file path to save the dataset.

    Returns:
        Dataset: The loaded dataset, or None if loading or saving fails.
    """
    ds = None
    try:
        ds = load_dataset(hf_ds_name)
        logger.info(f'Dataset {hf_ds_name} was loaded successfully!')
    except Exception as e:
        logger.warning(f"Error loading dataset {hf_ds_name}: {e}")
    try:
        ds.save_to_disk(ds_save_path)
        logger.info(f'Dataset was saved at {ds_save_path}!')
    except Exception as e:
        logger.warning(f"Error saving dataset to {ds_save_path}: {e}")
    return ds


def create_embeddings(hf_model_name: str, ds: Dataset, emb_save_path: str) -> np.ndarray:
    """
    Creates embeddings for the 'context' column of the given dataset using a specified model.

    Args:
        hf_model_name (str): The HuggingFace model name to use for embedding.
        ds (Dataset): The HuggingFace dataset to create embeddings from.
        emb_save_path (str): The file path to save the embeddings.

    Returns:
        np.ndarray: The generated context embeddings.
    """
    model = SentenceTransformer(hf_model_name).to(device)
    context_sentences = list(set(ds['train']['context']))
    context_embeddings = model.encode(context_sentences)
    np.save(emb_save_path, context_embeddings)
    logger.info(f'Embeddings were saved at {emb_save_path}')
    return context_embeddings


if __name__ == '__main__':
    # Get command-line arguments
    model_name = args.model
    dataset_name = args.dataset
    emb_save_path = args.emb_save_path
    ds_save_path = args.ds_save_path

    # Load and save the dataset
    ds = load_and_save_dataset(dataset_name, ds_save_path)

    # Create and save embeddings
    create_embeddings(model_name, ds, emb_save_path)
