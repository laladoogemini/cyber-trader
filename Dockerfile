# 🟢 修正點：升級到 Python 3.11，解決套件版本過新的問題
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統層級依賴 (編譯器與工具)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 複製需求清單
COPY requirements.txt .

# 安裝 Python 套件 (讓 pip 自動解決版本相容性)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 設定環境變數
ENV PORT=5000

# 啟動指令
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
