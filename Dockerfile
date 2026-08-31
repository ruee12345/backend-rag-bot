FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    make \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (for FAISS)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=1000 -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
