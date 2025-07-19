FROM python:3.10-slim

# Set environment variables early
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/tmp/nltk_data
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# Install system dependencies for PyMuPDF, Tesseract OCR, FAISS, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/nltk_data /tmp/hf_cache /tmp/dual_response_cache /tmp/textbooks /tmp/uploaded_pdfs && \
    chmod -R 777 /tmp

# Download NLTK data at build time (not runtime)
RUN python -c "import nltk; nltk.download('punkt', download_dir='/tmp/nltk_data'); nltk.download('punkt_tab', download_dir='/tmp/nltk_data'); nltk.download('stopwords', download_dir='/tmp/nltk_data')"

# Copy application code
COPY . .

# Expose the port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
