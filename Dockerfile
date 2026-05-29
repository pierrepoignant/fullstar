FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN apt-get update && apt-get install -y tzdata curl && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Bake the Epicure embeddings (core + siblings, ~13 MB total) into the image so
# the container starts instantly and never depends on Hugging Face at runtime.
RUN python -c "import chopper_recipes as cr; [cr.get_model(s) for s in cr.SIBLINGS]"
ENV HF_HUB_OFFLINE=1

EXPOSE 5000

# --preload loads the core model once in the master process; workers inherit it
# via copy-on-write instead of each re-loading it.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--preload", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]
