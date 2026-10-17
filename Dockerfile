FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# requirements.lock is intentionally dependency-free; keeping the copy makes
# the container build contract explicit and leaves room for a future pinned
# dependency without changing the build shape.
COPY requirements.txt requirements.lock ./

COPY . ./

EXPOSE 8000

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
