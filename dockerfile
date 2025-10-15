# Use official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY . .

# Install system dependencies (build tools + lib for numpy/pandas/etc.)
RUN apt-get update && apt-get install -y \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install certifi early so we can append our cert
RUN pip install --no-cache-dir certifi

# Copy your self-signed certificate into the container
COPY devplify.crt /usr/local/share/ca-certificates/devplify.crt

RUN update-ca-certificates \
    && cat /usr/local/share/ca-certificates/devplify.crt >> $(python3 -m certifi)

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8092

# Run the service
CMD ["python", "-m", "app.main"]
