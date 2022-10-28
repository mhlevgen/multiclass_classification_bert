FROM python:3.9-slim-buster

WORKDIR /python-docker

COPY docker_requirements.txt .
RUN pip install -r docker_requirements.txt

COPY . .

ENTRYPOINT ["python", "app.py"]