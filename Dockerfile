FROM python:3.10-slim

# 設定環境變數
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# 1. 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc g++ libxml2-dev libxslt-dev git \
    && rm -rf /var/lib/apt/lists/*

# 2. 安裝 Python 套件 (將所有套件寫在同一行，確保指令完整性)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel flask gunicorn requests openai markdown yfinance pandas==1.5.3 pandas_ta "numpy==1.23.5" lxml

# 3. 複製程式碼 (這必須是獨立的步驟！)
COPY . .

# 4. 暴露端口
ENV PORT=5000

# 5. 啟動指令
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 app:app
