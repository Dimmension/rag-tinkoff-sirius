import logging
import gradio as gr
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GradioBot(gr.Blocks):

    def __init__(self, title, css=None, theme=None):
        """Gradio интерфейс текстового бота

        Args:
            title (str): название интерфейса
            css (str, optional): css стили для интерфейса. Defaults to None.
            theme (gr.themes.Base, optional): gradio тема для интерфейса. Defaults to None.
        """
        super().__init__(
            title=title,
            css=css,
            theme=theme,
        )

        with self:
            def ask_question(question: str):
                if len(question.strip()) == 0:
                    gr.Warning("Строка не должна быть пустая!")
                    return None

                logging.info(f"Inputs: {question}.")

                package = {
                    "question": question
                }

                response = requests.post('http://backend_rag:7000/query', json=package)
                logging.info(f"Response status:\n{response}")

                if response.status_code != 200:
                    gr.Warning(f'Response status: {response.status_code}')
                    return None
                
                response = response.json()
                logging.info(f'ANSWER: {response}')
                return response['answer']

            gr.Markdown('<p style="font-size: 2.5em; text-align: center; margin-bottom: 1rem"><span style="font-family:Source Sans Pro; color:black"> Retrieval-Augmented Generation </span></p>')

            with gr.Tab("Поиск"):
                with gr.Row():
                    with gr.Column():
                        msg_txt = gr.Textbox(
                            label="Задайте вопрос:", 
                            placeholder="Как обрабатываются органические остатки?",
                            interactive=True,
                            show_copy_button=True,
                        )
                        ask_quest_btn = gr.Button(
                            value='Спросить', 
                            variant="primary"
                        )

                    with gr.Column():
                        answer_text_area = gr.TextArea(
                            label="Результат поиска:",
                            interactive=False,
                            lines=30,
                            show_copy_button=True,
                        )


                    msg_txt.submit(ask_question, msg_txt, answer_text_area)
                    ask_quest_btn.click(ask_question, msg_txt, answer_text_area)



def main():
    theme = gr.themes.Base(
        primary_hue="red",
        secondary_hue="red",
    )
    logging.info("Themes done.")
    demo = GradioBot(
        title='QA system',
        theme=theme,
    )
    logging.info("Bot done.")

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7000,
        share=False,
        show_error=True,
    )
    logging.info("Launch done.")


if __name__ == "__main__":
    logging.info("Starting gradio interface...")
    main()
    logging.info("Ready to go with gradio interface!")
