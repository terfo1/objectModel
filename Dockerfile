FROM python:3.10-slim

WORKDIR /app

COPY . /app
COPY best.pt /app/

RUN apt-get update && apt-get install -y ffmpeg \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
