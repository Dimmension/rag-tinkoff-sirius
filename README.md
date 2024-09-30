# Отборочное задание на смену по машинному обучению в Сириус

## Описание задачи:
В рамках отборочного задания требуется реализовать русскоязычную систему QA-систему. Основной задачей является построение модели, способной эффективно извлекать релевантную информацию и находить в вопросе соответствующие фрагменты.

## Цель:
Создать и продемонстрировать работоспособную систему для решения задачи вопросов-ответов (QA). Система должна быть развёрнута в виде телеграм-бота или веб-сервиса, доступного для взаимодействия, и демонстрировать корректные результаты на реальных примерах.

## Описание исходных данных
В качестве исходных данных представлен датасет **kuznetsoffandrey/sberquad** [https://huggingface.co/datasets/kuznetsoffandrey/sberquad].

**SberQuAD** — это датасет для задач QA, основанный на вопросах, созданных краудворкерами по набору статей из Википедии. Ответом на каждый вопрос является текстовый фрагмент из соответствующего отрывка статьи. Датасет является русскоязычным аналогом популярного англоязычного датасета SQuAD


![alt text](asstes/data_overview.png)

context - фрагмент из Википедии

question - вопрос к фрагменту

answers - ответ на вопрос, основанный на контексте

## Разработка решения

### Выбор решения:
Из предложенных решений было выбрано реализвать систему **Retrieval Augmented Generation(RAG)**, которая является одной из эффективнейших реализаций QA-систем.

1. Vector DB
Для хранения эмбеддингов была выбрана **ChromaDB** - легкая и понятная библиотека для создания векторной БД.

2. Retriever
В качестве retriever'a была выбрана вышедшая в Августе, 2024 года модель архитектуры sentence-transformers **ai-forever/ru-en-RoSBERTa**. Причина такого выбора, послужило, что она показала высокие метрики на сопоставлении контекст/вопрос. arxiv: https://arxiv.org/pdf/2408.12503

3. LLM
В качестве генеративной модели я взял модель **meta-llama-3.1-8b-instruct.Q6_K**, наиболее проверенную и надежную модель(работал с ней раньше). Важное уточнение, я использовал её квантизированную версию GGUF с помощью библиотеки llama.cpp: https://github.com/ggerganov/llama.cpp

4. API
Для создания API сервера был выбран FastAPI

5. Frontend
Для создания интерфейса был выбран Gradio

## Треборвания
- Docker
- Python3.10
- CUDA 11.8

## Установка
Установка модели LLaMA3.1 8B
```
cd src/app/models
wget https://huggingface.co/SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/meta-llama-3.1-8b-instruct.Q6_K.gguf
```

## Настройка 
Надо будет настроить IP под свои.
```
CHROMA_HOST = '172.22.100.166' # 127.0.0.1
APP_HOST = '127.0.0.1'
```

## Запуск
```
docker-compose up --build
```

Сервис будет доступен по: http://localhost:4840/

## Результы
Интерфейс gradio:

![alt text](asstes/gradio_result.png)

Затраты по памяти:

![alt text](asstes/memory_gpu.png)

## Контакты
По всем возникающим проблемам писать в тг: @dimmmension