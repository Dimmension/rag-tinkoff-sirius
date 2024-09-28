FROM python:3.8-slim

RUN apt-get -y update && apt-get -y upgrade

USER root

ENV VIRTUALENV=/home/root/venv
RUN python3 -m venv $VIRTUALENV
ENV PATH="$VIRTUALENV/bin:$PATH"

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt

WORKDIR /home/root/gradio

EXPOSE ${APP_PORT}

ENTRYPOINT [ "python", "app.py" ]
