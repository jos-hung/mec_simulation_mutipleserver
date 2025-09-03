FROM python:3.9.6
WORKDIR /app

RUN pwd

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY infer_service_proc.py ./src/infer_service_proc.py
COPY launch.py ./src/launch.py
COPY config.yaml ./src/config.yaml
COPY scheduler_proc.py ./src/scheduler_proc.py
COPY kill_all.py ./src/kill_all.py
COPY scripts ./src/scripts
COPY run.sh ./src/run.sh
COPY config.yaml ./src/config.yaml

EXPOSE 8000
