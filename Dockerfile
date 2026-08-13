FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD gunicorn -b 0.0.0.0:${PORT} -w 2 --threads 4 --timeout 120 server:app
# Local Flask (no Gunicorn): python server.py
