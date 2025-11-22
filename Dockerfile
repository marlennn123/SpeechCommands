FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /speech_app

COPY req.txt /app/
RUN pip install --upgrade pip && pip install -r req.txt
RUN pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html


COPY . /speech_app/


CMD ["uvicorn", "main:speech_app", "--host 0.0.0.0", "-port 8000"]





