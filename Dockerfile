# Multi-stage build to reduce final image size
FROM python:3.11-slim as builder

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install dependencies and download model
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2')"

# Final stage - minimal runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies and pre-downloaded model
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY . .

# Set environment variable to use cached model
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "vector_search_app:app", "--host", "0.0.0.0", "--port", "8080"]
