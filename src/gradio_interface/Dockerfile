FROM python:3.8-slim

RUN apt-get -y update && apt-get -y upgrade

USER root


ENV VIRTUALENV=/home/root/venv
RUN python3 -m venv $VIRTUALENV
ENV PATH="$VIRTUALENV/bin:$PATH"

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools
RUN pip install -r ./requirements.txt --no-cache-dir

WORKDIR /home/root/gradio_interface/

EXPOSE 7000
ENTRYPOINT ["python", "app.py"]
