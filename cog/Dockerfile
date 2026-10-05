FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && find /usr/local -name '__pycache__' -prune -exec rm -rf {} +
COPY predict.py server.py ./
COPY models models
USER nobody
EXPOSE 5000
CMD ["python", "server.py"]
