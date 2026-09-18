# ==========================================
# Stage 1: Build Dependencies
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Install C++ build tools required for C-extensions (rank-bm25, tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into wheels directory
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# Stage 2: Runtime Base
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application source code
COPY . /app

# Ensure non-buffered stdout/stderr logging
ENV PYTHONUNBUFFERED=1

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000
EXPOSE 8501

# Default command launches FastAPI API Gateway
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]