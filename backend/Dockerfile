FROM python:3.11-slim

# System env for Python (no buffering)
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Change working directory to /app
WORKDIR /app

# Create resources directory with appropriate permissions
RUN mkdir -p /app/resources && chmod 777 /app/resources

# Copy requirements first for better layer caching
COPY requirements.txt ./

# Create venv and install requirements with verbose logging
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the entrypoint script first and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy everything except resources directory
COPY . .
RUN rm -rf resources

EXPOSE 8005

ENTRYPOINT ["/app/entrypoint.sh"]