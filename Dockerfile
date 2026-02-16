FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# minimal system deps for Pillow (jpeg/png + zlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code + model
COPY src ./src
COPY app ./app
COPY models ./models

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]