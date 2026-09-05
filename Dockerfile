FROM python:3.13-slim AS deps

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM deps AS app

COPY src/ ./src/

CMD ["python", "src/data_generation.py"]
