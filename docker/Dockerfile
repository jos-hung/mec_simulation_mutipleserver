FROM python:3.9.6
WORKDIR /app

RUN pwd

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY *.py ./src/
COPY *.yaml ./src/
COPY *.txt ./src/
COPY *.sh ./src/
EXPOSE 10000
