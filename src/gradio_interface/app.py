import logging
import gradio as gr
import requests

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GradioBot(gr.Blocks):
    def __init__(self, title, css=None, theme=None):
        """
        Gradio interface for a text-based bot.

        Args:
            title (str): The title of the interface.
            css (str, optional): CSS styles for the interface. Defaults to None.
            theme (gr.themes.Base, optional): Gradio theme for the interface. Defaults to None.
        """
        super().__init__(title=title, css=css, theme=theme)

        # Build the Gradio interface
        with self:
            def ask_question(question: str):
                # Validate input question
                if len(question.strip()) == 0:
                    gr.Warning("Строка не должна быть пустая!")  # Warning for empty input
                    return None

                logging.info(f"Input question: {question}.")

                # Prepare the request payload
                package = {
                    "question": question
                }

                # Send request to the backend API
                response = requests.post('http://backend_rag:7000/query', json=package)
                logging.info(f"Response status code: {response.status_code}")

                if response.status_code != 200:
                    gr.Warning(f'Ошибка ответа: {response.status_code}')  # Warning for failed API response
                    return None
                
                # Parse and return the API response
                response = response.json()
                logging.info(f'Ответ: {response}')
                return response['answer']

            # Header for the interface
            gr.Markdown(
                '<p style="font-size: 2.5em; text-align: center; margin-bottom: 1rem">'
                '<span style="font-family:Source Sans Pro; color:black">'
                'Retrieval-Augmented Generation</span></p>'
            )

            # Search Tab
            with gr.Tab("Поиск"):
                with gr.Row():
                    with gr.Column():
                        # Textbox for user input
                        msg_txt = gr.Textbox(
                            label="Задайте вопрос:",
                            placeholder="Как обрабатываются органические остатки?",
                            interactive=True,
                            show_copy_button=True,
                        )
                        # Button to trigger the question
                        ask_quest_btn = gr.Button(
                            value='Спросить',
                            variant="primary"
                        )

                    with gr.Column():
                        # TextArea to display the answer
                        answer_text_area = gr.TextArea(
                            label="Результат поиска:",
                            interactive=False,
                            lines=30,
                            show_copy_button=True,
                        )

                    # Define event handling
                    msg_txt.submit(ask_question, msg_txt, answer_text_area)
                    ask_quest_btn.click(ask_question, msg_txt, answer_text_area)


def main():
    # Define the theme for the Gradio app
    theme = gr.themes.Base(
        primary_hue="red",
        secondary_hue="red",
    )
    logging.info("Theme setup complete.")

    # Initialize the GradioBot with the specified title and theme
    demo = GradioBot(
        title='QA system',
        theme=theme,
    )
    logging.info("Bot initialization complete.")

    # Launch the Gradio app
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7000,
        share=False,
        show_error=True,
    )
    logging.info("App launched successfully.")


# Entry point for starting the app
if __name__ == "__main__":
    logging.info("Starting Gradio interface...")
    main()
    logging.info("Gradio interface is ready!")