import os
import requests
import logging
import pickle
import gradio as gr
from dotenv import load_dotenv


load_dotenv()
logging.root.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

APP_HOST_IP = "0.0.0.0"
GRADIO_APP_PORT = int(os.getenv('GRADIO_APP_PORT'))
RAG_URL = f"http://rag_app:7000/get_answer/"
RAG_UPDATE_CORPUS = f"http://rag_app:7000/update_corpus/"
CORPUS_FILE_PATH = os.getenv('CORPUS_FILE_PATH')
logging.info(f'CORPUS_FILE_PATH: {CORPUS_FILE_PATH};\tAPP_HOST_IP: {APP_HOST_IP};\tAPP_PORT: {GRADIO_APP_PORT};\tRAG_URL: {RAG_URL}')

theme = gr.themes.Base(
    primary_hue="red",
    secondary_hue="red",
)

CORPUS_INFO = ""


def print_directory_tree(root_dir, indent=""):
    items = sorted(os.listdir(root_dir))
    for index, item in enumerate(items):
        path = os.path.join(root_dir, item)
        if index == len(items) - 1:
            connector = "└──"
        else:
            connector = "├──"
        
        logging.info(indent + connector + item)
        
        if os.path.isdir(path):
            if index == len(items) - 1:
                new_indent = indent + "    "
            else:
                new_indent = indent + "│   "
            print_directory_tree(path, new_indent)


print_directory_tree(os.getcwd())

try:
    with open(CORPUS_FILE_PATH, 'rb') as f:
        corpus = pickle.load(file=f)

    for i, el in enumerate(corpus):
        CORPUS_INFO += f"({i}). {el}\n\n"
except Exception:
     logging.warning(f"Dont have any corpus for RAG!")


def process_file(file):
    global CORPUS_INFO
    CORPUS_INFO = ""
    with open(file.name, "rb") as f:
        content = pickle.load(f)
    # Здесь можно добавить любую дополнительную обработку файла
    for i, el in enumerate(content):
        CORPUS_INFO += f"({i + 1}) {el}\n\n"
    with open(CORPUS_FILE_PATH, "wb") as f:
        pickle.dump(content, f)

    logging.info(f"Success write to file")
    response = requests.get(RAG_UPDATE_CORPUS)
    logging.info(f"Try update RAG corpus with code: {response.status_code}")

    return f"Содержимое файла:\n\n{CORPUS_INFO}", CORPUS_INFO


with gr.Blocks(title="Retrieval-Augmented Generation", theme=theme) as demo:

    def ask_question(question, confidience, answer_mode, discard_mode):
        global CORPUS_INFO
        if len(question.strip()) == 0:
            gr.Warning("Строка не должна быть пустая!")
            return None

        if len(CORPUS_INFO) == 0:
            gr.Warning("Настройте базу знаний, на данный момент ничего не найдено!")
            return None

        logging.info(f"Inputs: {question}; {confidience}; {answer_mode}; {discard_mode}.")

        if answer_mode == 'SOFT':
            is_soft_answer = True
        else:
            is_soft_answer = False

        if discard_mode == 'SOFT':
            is_soft_discard = True
        else:
            is_soft_discard = False
        
        package = {
            "query": question,
            "threshold_confidience": confidience,
            "is_soft_answer": is_soft_answer,
            "is_soft_discard": is_soft_discard,
        }

        response = requests.post(RAG_URL, json=package)
        logging.info(f"Response status:\n{response}")

        if response.status_code != 200:
            gr.Warning(f'Response status: {response.status_code}')
            return None, None

        response = response.json()

        return response['answer'], response['retrieve_logs']


    gr.Markdown(
                '<p style="font-size: 2.5em; text-align: center; margin-bottom: 1rem"><span style="font-family:Source Sans Pro; color:black"> Retrieval-Augmented Generation </span></p>'
            )

    with gr.Tab("Поиск"):

        with gr.Row():

            with gr.Column():
                msg_txt = gr.Textbox(
                    label="Задайте вопрос:", 
                    placeholder="Как устроена технологическая цепочка обработки документов?",
                    interactive=True,
                    show_copy_button=True,
                )
                ask_quest_btn = gr.Button(
                    value='Спросить', 
                    variant="primary"
                )
                confidience_slider = gr.Slider(
                    label="Минимальная степень уверенности, при которой RAG считает, что на заданный вопрос существует соответствующий ответ в базе знаний.",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.01,
                    interactive=True,
                )
                answer_modes_radio = gr.Radio(
                    label="Режим ПОИСКА ОТВЕТА:\"SOFT\" -> при генерации ответа учитываются знания по ТОП-3 кандидатам;\n\n\"STRICT\" -> ответ исключительно по ТОП-1.",
                    choices=["SOFT", "STRICT"],
                    value="SOFT",
                    interactive=True,
                )
                discard_modes_radio = gr.Radio(
                    label="Режим генерации ОШИБКИ ПОИСКА ОТВЕТА:\"SOFT\" -> допускается небольшие самовольные мысли и советы;\n\n\"STRICT\" -> ответ исключительно по базе, никаких самовольностей.",
                    choices=["SOFT", "STRICT"],
                    value="SOFT",
                    interactive=True,
                )
                retrieve_text_area = gr.TextArea(
                    label="Результаты поиска:",
                    interactive=False,
                    lines=12,
                    show_copy_button=True,
                )

            with gr.Column():
                answer_text_area = gr.TextArea(
                    label="Результат поиска:",
                    interactive=False,
                    lines=30,
                    show_copy_button=True,
                )

            with gr.Column():
                corpus_text_area = gr.TextArea(
                    label="База знаний:",
                    value=CORPUS_INFO,
                    interactive=False,
                    lines=30,
                    show_copy_button=True,
                )

            msg_txt.submit(ask_question, [msg_txt, confidience_slider, answer_modes_radio, discard_modes_radio], [answer_text_area, retrieve_text_area])
            ask_quest_btn.click(ask_question, [msg_txt, confidience_slider, answer_modes_radio, discard_modes_radio], [answer_text_area, retrieve_text_area])

    with gr.Tab("Настройка Базы Знаний"):
        
        with gr.Row():

            with gr.Column():

                instr_area = gr.TextArea(
                    label="ИНСТРУКЦИЯ",
                    interactive=False,
                    lines=12,
                    show_copy_button=True,
                    value="""1) При старте приложения подключается последняя загруженная база знаний (БЗ).
2) Ожидаемый формат файла БЗ => PICKLE файл, в котором закодирован **список** из кандидатов.
3) Загрузите файл в соответствующем окне и нажмите на кнопку \"Обработать файл\".
4) После обработки файла его содержание можно увидеть в окне ниже.
5) После обработки файла новая БЗ автоматически подключается к сервису RAG и дальнейший поиск работает через новую БЗ.""",
                )

            with gr.Column():

                # тут компонент файла
                file_input = gr.File(
                    label="Загрузить файл",
                    type="filepath"
                )

                file_output = gr.TextArea(
                    label="Результат обработки файла",
                    interactive=False,
                    lines=12,
                )

                process_button = gr.Button("Обработать файл")

                process_button.click(
                    fn=process_file,
                    inputs=file_input,
                    outputs=[file_output, corpus_text_area]
                )

                # corpus_text_area.update()

demo.launch(
    server_name=APP_HOST_IP,
    server_port=GRADIO_APP_PORT,
    share=True,
    show_error=True,
)
