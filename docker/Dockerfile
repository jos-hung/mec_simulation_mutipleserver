FROM python:3.9.6
WORKDIR /app

RUN pwd

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY infer_service_proc.py ./src/infer_service_proc.py
COPY launch.py ./src/launch.py
COPY config.yaml ./src/config.yaml
COPY scheduler_proc.py ./src/scheduler_proc.py
COPY kill_all.py ./src/kill_all.py
COPY scripts ./src/scripts
COPY run.sh ./src/run.sh
COPY config.yaml ./src/config.yaml
COPY .git ./src/.git
COPY post_from_host.py ./src/post_from_host.py
COPY inference_request.py ./src/inference_request.py

EXPOSE 10000
