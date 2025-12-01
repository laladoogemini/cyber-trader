# 使用 Python 3.11 (最穩定版本)
FROM python:3.11-slim

# 設定環境變數
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 設定工作目錄
WORKDIR /app

# 1. 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc g++ libxml2-dev libxslt-dev git \
    && rm -rf /var/lib/apt/lists/*

# 2. 直接安裝 Python 套件 (不再讀取 requirements.txt，避開格式錯誤)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    setuptools wheel \
    flask gunicorn requests \
    openai markdown \
    yfinance pandas pandas_ta lxml \
    "numpy<2.0.0"

# 3. 複製程式碼
COPY . .

# 4. 暴露端口
ENV PORT=5000

# 5. 啟動指令
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
