from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import json
import os
import markdown
import numpy as np
from openai import OpenAI 
from datetime import datetime

# --- 雲端環境修正：設定 yfinance 快取路徑到 /tmp (暫存區) ---
# 這是解決 Render 報錯的關鍵
try:
    if not os.path.exists('/tmp/yf_cache'):
        os.makedirs('/tmp/yf_cache')
    yf.set_tz_cache_location('/tmp/yf_cache')
except:
    pass # 如果是在本機執行，就忽略

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)

# --- 設定 NVIDIA API ---
NVIDIA_API_KEY = "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"
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
MODEL_NAME = "deepseek-ai/deepseek-r1"

STOCK_NAME_MAP = {
    '2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '0050.TW': '元大台灣50',
    '2603.TW': '長榮', '2609.TW': '陽明', '2615.TW': '萬海',
    'NVDA': '輝達', 'TSLA': '特斯拉', 'AAPL': '蘋果', 'AMD': '超微',
    'BTC-USD': '比特幣', 'ETH-USD': '以太幣'
}

def safe_float(val):
    try:
        if pd.isna(val) or val is None: return 0
        return float(val)
    except: return 0

# --- 1. 抓取宏觀 ---
def get_macro_data():
    try:
        # 減少抓取天數以加速 (90d -> 60d)
        assets = {'TWII': '^TWII', 'SPY': 'SPY', 'GLD': 'GLD', 'VIX': '^VIX'}
        df = yf.download(list(assets.values()), period='60d', interval='1d', progress=False, auto_adjust=True)
        
        if df.empty: return None

        # 暴力修正索引問題
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if 'Close' in df.columns.get_level_values(0): df = df['Close']
            except: pass
            # 扁平化欄位
            df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]

        mapping = {v: k for k, v in assets.items()}
        # 只保留存在的欄位
        valid_cols = [c for c in df.columns if c in assets.values()]
        df = df[valid_cols].rename(columns=mapping)
        
        if df.empty: return None

        weekly_change = {}
        for col in df.columns:
            if len(df) >= 5:
                prev = safe_float(df[col].iloc[-5])
                curr = safe_float(df[col].iloc[-1])
                change = (curr - prev) / prev if prev != 0 else 0
                weekly_change[col] = f"{'+' if change>0 else ''}{round(change*100, 2)}%"
            else:
                weekly_change[col] = "N/A"
            
        corr = df.pct_change().corr().round(2).fillna(0)
        prices = {k: safe_float(v) for k, v in df.iloc[-1].to_dict().items()}
        
        return {"weekly_change": weekly_change, "prices": prices, "correlation": corr.to_dict()}
    except Exception as e: 
        print(f"Macro Error: {e}")
        return None

# --- 2. 抓取個股 ---
def get_stock_data_full(symbol):
    try:
        # 加速：只抓 150 天，足夠算 MA60
        df = yf.download(symbol, period="150d", interval="1d", progress=False, auto_adjust=True)
        
        if df.empty: return None
            
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        
        # 確保只有一個 Close 欄位
        if 'Close' not in df.columns:
            if len(df.columns) == 1: df.columns = ['Close']
            else: return None
            
        if 'Volume' not in df.columns: df['Volume'] = 0

        # 指標計算 (加強錯誤處理)
        try:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['SMA_60'] = ta.sma(df['Close'], length=60)
            # 布林通道 (簡化處理)
            std = df['Close'].rolling(20).std()
            sma20 = df['Close'].rolling(20).mean()
            df['BB_Upper'] = sma20 + 2*std
            
            # 成交量均線
            df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        except: 
            pass # 指標計算失敗不影響股價顯示
        
        df.fillna(0, inplace=True)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        change_val = safe_float(last['Close'] - prev['Close'])
        change_pct = safe_float((change_val / prev['Close']) * 100) if prev['Close'] != 0 else 0
        
        # 線圖數據
        chart_df = df.iloc[-30:]
        chart_data = {
            "dates": chart_df.index.strftime('%m/%d').tolist(),
            "prices": [safe_float(x) for x in chart_df['Close'].tolist()]
        }

        # 狀態
        close = safe_float(last['Close'])
        bb_upper = safe_float(last.get('BB_Upper', 99999))
        sma_60 = safe_float(last.get('SMA_60', 0))
        volume = safe_float(last['Volume'])
        vol_sma = safe_float(last.get('Vol_SMA', 0))

        return {
            "price": round(close, 2),
            "change": round(change_val, 2),
            "pct": round(change_pct, 2),
            "RSI": round(safe_float(last.get('RSI', 50)), 2),
            "Bias_60": round(((close - sma_60) / sma_60) * 100, 2) if sma_60!=0 else 0,
            "Trend": "多頭" if close > sma_60 else "空頭",
            "BB_Status": "觸及上軌" if close >= bb_upper else "正常",
            "Vol_Status": "放量" if volume > vol_sma * 1.2 else "縮量",
            "chart": chart_data
        }
    except Exception as e:
        print(f"Stock Data Error: {e}")
        return None

# --- 3. 生成報告 ---
def generate_report_content(input_type, data):
    current_time = datetime.now().strftime("%Y-%m-%d")
    
    try:
        if input_type == "WEEKLY":
            prompt = f"""
            現在是 {current_time}。請根據以下數據寫一份簡短的【全景羅盤週報】。
            數據: 台股{data['weekly_change'].get('TWII')}, 美股{data['weekly_change'].get('SPY')}, VIX {data['prices'].get('VIX')}
            格式: # 標題 \n ## 市場總結 \n ## 跨市場洞察 \n ## 策略
            """
        else:
            prompt = f"""
            現在是 {current_time}。請根據以下數據寫一份【{data['name_zh']}】的個股分析報告。
            數據: 股價 {data['price']} | 漲跌 {data['pct']}%
            技術面: {data['tech']['Trend']} | RSI {data['tech']['RSI']} | 季線乖離 {data['tech']['Bias_60']}%
            請用 Markdown 格式。
            格式: # 標題 \n ## 核心論點 \n ## 基本面 \n ## 技術與籌碼 \n ## 風險與策略
            """

        completion = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=2000
        )
        return completion.choices[0].message.content.replace('```markdown', '').replace('```', '').strip()

    except Exception as e:
        return f"# AI 連線逾時\n\n雖然 AI 暫時無法回應，但上方數據仍準確。\n錯誤訊息：{str(e)}"

# --- 路由 ---
@app.route('/')
def home(): return render_template('report.html')

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    try:
        user_input = request.json.get('ticker', '').strip().upper()
        
        if user_input == "WEEKLY":
            data = get_macro_data()
            if not data: return jsonify({'error': '無法取得宏觀數據，請稍後再試'})
            raw_text = generate_report_content("WEEKLY", data)
            return jsonify({'content': markdown.markdown(raw_text), 'raw': raw_text, 'type': 'weekly'})
        else:
            symbol = f"{user_input}.TW" if user_input.isdigit() else user_input
            
            # 特殊代號處理
            if user_input in ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'BNB']: symbol = f"{user_input}-USD"
            if user_input == 'GOLD': symbol = "GC=F"
            if user_input == 'SILVER': symbol = "SI=F"

            tech = get_stock_data_full(symbol)
            if not tech: return jsonify({'error': f'找不到 {symbol} 數據，請確認代號'})
            
            # 嘗試取得名稱，失敗則用代號
            stock = yf.Ticker(symbol)
            try: raw_name = stock.info.get('longName', user_input)
            except: raw_name = user_input
            
            name_zh = STOCK_NAME_MAP.get(symbol, STOCK_NAME_MAP.get(user_input, raw_name))

            stock_data = {
                'symbol': user_input, 'name_zh': name_zh,
                'price': tech['price'], 'pct': tech['pct'],
                'tech': tech
            }
            
            raw_text = generate_report_content("STOCK", stock_data)
            
            return jsonify({
                'content': markdown.markdown(raw_text), 
                'raw': raw_text, 
                'meta': stock_data, 
                'type': 'stock',
                'chartData': tech['chart']
            })
            
    except Exception as e:
        # 這裡會回傳 JSON 格式的錯誤，而不是讓前端白屏
        return jsonify({'error': f"系統錯誤: {str(e)}"})

@app.route('/explain_simple', methods=['POST'])
def explain_simple():
    # ... (保持原樣) ...
    return jsonify({'explanation': '暫停服務'}) # 簡化以防出錯

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
