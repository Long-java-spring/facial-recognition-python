FROM python:3.6-slim-buster
LABEL maintainer="Autobot Asia<ai@autobot.asia>"

RUN apt-get update && apt-get -y install libsm6 libxext6 libxrender-dev libglib2.0 libgl1-mesa-dev; apt-get clean
RUN pip install opencv-contrib-python-headless

COPY requirements.txt /app/requirements.txt
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

# add entrypoint.sh
RUN chmod +x ./entrypoint.sh
