import os
import logging
import torch
from torch import cuda 
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

load_dotenv()
CUDA_NUMBER = os.getenv('CUDA_NUMBER')
RETRIEVER_MODEL_NAME = os.getenv('RETRIEVER_MODEL_NAME')
RERANKER_MODEL_NAME = os.getenv('RERANKER_MODEL_NAME')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')
LLM_DTYPE = torch.bfloat16 if os.getenv('LLM_DTYPE') == 'bfloat' else torch.float16
MODEL_CACHE = os.getenv('MODEL_CACHE')
CORPUS_FILE_PATH = os.getenv('CORPUS_FILE_PATH')
READ_HF_TOKEN = os.getenv('READ_HF_TOKEN')

HARD_SYSTEM_PROMPT = """Ты виртуальный помощник Валера, отвечающий на вопросы связанные с темой \"Единая автоматизированная система документооборота (ЕАСД). Российские Железные Дороги (РЖД)\". Ты получаешь от пользователя пару вопрос (QUESTION) и ответ на этот вопрос (ANSWER), используя только эту полезную информацию (QUESTION) и (ANSWER), и без дополнительных пояснений и прочих твоих мыслей, напиши ответ от лица Валеры, пожалуйста!""" # strict
SOFT_SYSTEM_PROMPT = """Ты виртуальный помощник Валера, отвечающий на вопросы связанные с темой \"Единая автоматизированная система документооборота (ЕАСД). Российские Железные Дороги (РЖД)\". Ты получаешь от пользователя вопрос (QUESTION) и 3 наиболее подходящих ответа на этот вопрос (ANSWER), используя только эту полезную информацию (QUESTION), (ANSWER_1), (ANSWER_2), (ANSWER_3), и без дополнительных пояснений и специальных меток (ANSWER) и прочие, напиши ответ от лица Валеры, пожалуйста!""" # soft
USER_PROMPT = """
(QUESTION) = \"{}\"
(ANSWER) = \"{}\"
"""
SOFT_USER_PROMPT = """
(QUESTION) = \"{}\"
(ANSWER_1) = \"{}\"
(ANSWER_2) = \"{}\"
(ANSWER_3) = \"{}\"
"""
HARD_DISCARD_SYS_PROMPT = """Ты виртуальный помощник Валера, отвечающий на вопросы связанные с только темой \"Единая автоматизированная система документооборота (ЕАСД). Российские Железные Дороги (РЖД)\". Ты получаешь от пользователя вопрос, на который ты не можешь ответить, потому что в базе знаний нет точного ответа на этот вопрос. Вежливо извинись и упомяни эту проблему, если это вопрос, который связан с РЖД, тогда можешь пояснить свой ответ, если вопрос не связан с РЖД, то не надо никаких дополнительных пояснений, не пытайся ответить вовсе. Пиши от лица Валеры, пожалуйста!""" # - strict
SOFT_DISCARD_SYS_PROMPT = """Ты виртуальный помощник Валера, отвечающий на вопросы связанные с только темой \"Единая автоматизированная система документооборота (ЕАСД). Российские Железные Дороги (РЖД)\". Ты получаешь от пользователя вопрос, на который ты не можешь ответить, потому что в базе знаний нет точного ответа на этот вопрос. Вежливо извинись и упомяни эту проблему, и попроси переформулировать вопрос, от лица Валеры, пожалуйста!""" # - soft
DISCARD_USER_PROMPT = """
Вопрос: \"{}\"
"""

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_NUMBER
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

logging.info(f"CUDA_NUMBER: {CUDA_NUMBER}; DEVICE: {DEVICE}")
logging.info(f"HuggingFace`s token: {READ_HF_TOKEN}")
logging.info(f"Models: {RETRIEVER_MODEL_NAME}; {RERANKER_MODEL_NAME}; {LLM_MODEL_NAME} (with dtype {LLM_DTYPE}); in cache {MODEL_CACHE}")


ai-forever/ru-en-RoSBERTa