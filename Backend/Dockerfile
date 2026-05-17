# ================================================================
# Stage 1: Build & Dependencies
# ================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Tạo môi trường ảo (venv)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Cài đặt thư viện Python vào trong venv
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ================================================================
# Stage 2: Runtime
# ================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Cài đặt CÁC THƯ VIỆN HỆ THỐNG cần cho runtime (trực tiếp tại đây cho an toàn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy toàn bộ môi trường ảo (đã chứa các thư viện Python) từ builder sang
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY . .

RUN mkdir -p /app/models /app/data

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]