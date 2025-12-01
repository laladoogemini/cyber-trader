FROM python:3.11-slim

# 設定環境變數：防止 Python 產生 .pyc 檔，並讓 log 直接輸出
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# 安裝系統工具 (一次裝完)
RUN apt-get update && apt-get install -y \
    gcc g++ libxml2-dev libxslt-dev git \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 暴露端口
ENV PORT=5000

# 啟動指令 (增加 workers 提升穩定度)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
