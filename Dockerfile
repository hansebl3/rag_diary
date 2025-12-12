FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if any needed for pymysql or chroma)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables will be overridden by docker-compose
ENV OLLAMA_BASE_URL="http://100.65.53.9:11434"
ENV DB_HOST="127.0.0.1"

# Command to run the app
CMD ["streamlit", "run", "src/app.py", "--server.port", "8510", "--server.address", "0.0.0.0"]
