FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DECK_VISION_CACHE_DIR=/var/cache/deck-vision

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin deckvision

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY data /app/data

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[server]" \
    && mkdir -p /var/cache/deck-vision \
    && chown -R deckvision:deckvision /var/cache/deck-vision /app

USER deckvision

EXPOSE 8000

CMD ["uvicorn", "deck_vision.server:app", "--host", "0.0.0.0", "--port", "8000"]
