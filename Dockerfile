# Use official Python runtime as a parent image
FROM python:3.13-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     build-essential \
#     libopencv-dev \
#     python3-opencv \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    -o Debug::pkgProblemResolver=yes \
    -o APT::Get::Show-Versions=yes \
    -o Acquire::http::Timeout=60 \
    -o Acquire::ftp::Timeout=60 \
    gcc g++ build-essential libopencv-dev python3-opencv \
    && rm -rf /var/lib/apt/lists/*


# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port for FastAPI
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.fastapi.app:app", "--host", "0.0.0.0", "--port", "8000"]
