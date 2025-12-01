# 使用輕量級的 Python 官方映像檔
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統層級的依賴 (這是 yfinance 和 pandas 需要的)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# 複製需求清單
COPY requirements.txt .

# 關鍵指令：使用 --prefer-binary 強制下載「已編譯版」，不佔用記憶體
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# 複製所有程式碼
COPY . .

# 設定環境變數
ENV PORT=5000

# 啟動指令 (設定 120秒超時，防止 AI 思考太久被斷線)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]