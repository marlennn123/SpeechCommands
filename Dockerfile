FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Устанавливаем системные библиотеки (ТОЛЬКО необходимые)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        gcc \
        g++ \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# Устанавливаем PyTorch CPU Only (очень важно!)
RUN pip install --no-cache-dir torch==2.2.2+cpu \
      -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir torchaudio==2.2.2+cpu \
      -f https://download.pytorch.org/whl/torch_stable.html

# Устанавливаем оставшиеся зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY app.py .
COPY audio_model.pth .
COPY labels.pth .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]