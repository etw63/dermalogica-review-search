# Multi-stage build to reduce final image size
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage - minimal runtime image
FROM python:3.11-slim

# Copy only the installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY vector_search_app.py .
COPY templates/ templates/
COPY static/ static/

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /home/app
USER app

# Set working directory
WORKDIR /home/app

# Copy application files to user directory
COPY --chown=app:app vector_search_app.py .
COPY --chown=app:app templates/ templates/
COPY --chown=app:app static/ static/

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "vector_search_app.py"]
