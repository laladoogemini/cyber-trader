# 🟢 修正點：升級到 Python 3.12 (滿足套件的強制要求)
FROM python:3.12-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統層級依賴
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 複製需求清單
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 設定環境變數
ENV PORT=5000

# 啟動指令
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
