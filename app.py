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

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)

# --- 設定 NVIDIA API ---
# 請確認此 Key 是否仍有效
NVIDIA_API_KEY = "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"
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
MODEL_NAME = "deepseek-ai/deepseek-r1"

STOCK_NAME_MAP = {
    # 台股
    '2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '0050.TW': '元大台灣50',
    '2603.TW': '長榮', '2609.TW': '陽明', '2615.TW': '萬海',
    # 美股
    'NVDA': '輝達 (NVIDIA)', 'TSLA': '特斯拉 (Tesla)', 'AAPL': '蘋果 (Apple)', 'AMD': '超微 (AMD)',
    # 加密貨幣 (必須加 -USD)
    'BTC-USD': '比特幣 (Bitcoin)',
    'ETH-USD': '以太幣 (Ethereum)',
    'SOL-USD': '索拉納 (Solana)',
    'ADA-USD': '艾達幣 (Cardano)',
    'XRP-USD': '瑞波幣 (Ripple)',
    'BNB-USD': '幣安幣 (BNB)',
    # 貴金屬 (期貨代號)
    'GC=F': '黃金期貨 (Gold)',
    'SI=F': '白銀期貨 (Silver)'
}

def safe_float(val):
    try:
        if pd.isna(val) or val is None: return 0
        return float(val)
    except: return 0

# --- 1. 抓取宏觀 ---
def get_macro_data():
    try:
        print("正在抓取宏觀數據...")
        assets = {'TWII': '^TWII', 'SPY': 'SPY', 'GLD': 'GLD', 'VIX': '^VIX'}
        df = yf.download(list(assets.values()), period='90d', interval='1d', progress=False, auto_adjust=True)
        if df.empty: return None

        # 索引修正
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0): df = df['Close']
                df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
        except: pass

        mapping = {v: k for k, v in assets.items()}
        df.rename(columns=mapping, inplace=True)
        
        available_cols = [c for c in df.columns if c in assets.keys()]
        if not available_cols: return None

        weekly_change = {}
        for col in available_cols:
            if len(df) >= 5:
                prev = safe_float(df[col].iloc[-5])
                curr = safe_float(df[col].iloc[-1])
                change = (curr - prev) / prev if prev != 0 else 0
                weekly_change[col] = f"{'+' if change>0 else ''}{round(change*100, 2)}%"
            else:
                weekly_change[col] = "N/A"
            
        corr = df[available_cols].pct_change().corr().round(2).fillna(0)
        prices = {k: safe_float(v) for k, v in df.iloc[-1].to_dict().items() if k in available_cols}
        
        return {"weekly_change": weekly_change, "prices": prices, "correlation": corr.to_dict()}
    except Exception as e: 
        print(f"宏觀數據錯誤: {e}")
        return None

# --- 2. 抓取個股 (含圖表數據) ---
def get_stock_data_full(symbol):
    print(f"獲取個股數據: {symbol}...")
    try:
        df = yf.download(symbol, period="200d", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
            
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if 'Close' not in df.columns:
            if len(df.columns) == 1: df.columns = ['Close']
            else: return None
        if 'Volume' not in df.columns: df['Volume'] = 0

        # 計算指標
        try:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_60'] = ta.sma(df['Close'], length=60)
            bbands = ta.bbands(df['Close'], length=20, std=2)
            if bbands is not None:
                df['BB_Upper'] = bbands[bbands.columns[2]]
                df['BB_Lower'] = bbands[bbands.columns[0]]
            # 成交量均線
            df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
        except: pass
        
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        change_val = safe_float(last['Close'] - prev['Close'])
        change_pct = safe_float((change_val / prev['Close']) * 100) if prev['Close'] != 0 else 0
        
        chart_df = df.iloc[-30:]
        chart_data = {
            "dates": chart_df.index.strftime('%m/%d').tolist(),
            "prices": [safe_float(x) for x in chart_df['Close'].tolist()]
        }

        # --- 狀態判斷 (準備給 AI 的素材) ---
        bb_upper = safe_float(last.get('BB_Upper', 99999))
        bb_lower = safe_float(last.get('BB_Lower', 0))
        sma_20 = safe_float(last.get('SMA_20', 0))
        sma_60 = safe_float(last.get('SMA_60', 0))
        close = safe_float(last['Close'])
        volume = safe_float(last['Volume'])
        vol_sma = safe_float(last.get('Vol_SMA', 0))
        
        bb_status = "正常"
        if close >= bb_upper: bb_status = "突破布林上軌(強勢/過熱)"
        elif close <= bb_lower: bb_status = "跌破布林下軌(弱勢/超賣)"
        
        trend = "盤整"
        if close > sma_20 > sma_60: trend = "多頭排列 (強)"
        elif close < sma_20 < sma_60: trend = "空頭排列 (弱)"
        
        vol_status = "放量" if volume > vol_sma * 1.2 else "縮量" if volume < vol_sma * 0.8 else "量能平穩"

        return {
            "price": round(close, 2),
            "change": round(change_val, 2),
            "pct": round(change_pct, 2),
            "volume": int(volume),
            "RSI": round(safe_float(last.get('RSI', 50)), 2),
            "Bias_60": round(((close - sma_60) / sma_60) * 100, 2) if sma_60 != 0 else 0,
            "Trend": trend,
            "BB_Status": bb_status,
            "Vol_Status": vol_status,
            "chart": chart_data
        }
    except Exception as e:
        print(f"個股數據處理錯誤: {e}")
        return None

# --- 3. 生成報告 (關鍵修正：強制時空定位) ---
def generate_report_content(input_type, data):
    # 獲取電腦當前時間，這是最真實的 "現在"
    current_time = datetime.now().strftime("%Y年%m月%d日")
    
    print(f"AI 撰寫中... (基準日期: {current_time})")
    
    if input_type == "WEEKLY":
        prompt = f"""
        現在時間是：【{current_time}】。
        你是一位宏觀分析師。請根據「提供的數據」撰寫【全景羅盤週報】。
        
        ⚠️ 重要指令：
        1. 你的訓練資料可能過時，請「完全忽略」你記憶中的舊數據。
        2. 一切以我提供的【真實數據】為準。
        
        【真實數據】
        日期: {current_time}
        台股(TWII) 本週表現: {data['weekly_change'].get('TWII', 'N/A')} (現價: {data['prices'].get('TWII', 'N/A')})
        美股(SPY) 本週表現: {data['weekly_change'].get('SPY', 'N/A')} (現價: {data['prices'].get('SPY', 'N/A')})
        恐慌指數(VIX): {data['prices'].get('VIX', 'N/A')} (若>20代表恐慌, <15代表貪婪)
        相關性: 美股與黃金相關係數 {data['correlation'].get('SPY', {}).get('GLD', 'N/A')} (正值同向, 負值反向)
        
        Markdown格式: # 標題 \n ## 市場總結 \n ## 跨市場洞察 \n ## 策略建議
        """
    else:
        # 強制注入「現在」的數據意義
        rsi_val = data['tech']['RSI']
        rsi_desc = "超買區(過熱)" if rsi_val > 70 else "超賣區(低檔)" if rsi_val < 30 else "中性區"
        
        prompt = f"""
        現在時間是：【{current_time}】。
        你是一位專業操盤手。請根據以下【即時數據】撰寫【{data['name_zh']} ({data['symbol']})】的深度分析報告。

        ⚠️ 絕對準則：
        1. 這是 {current_time} 的最新盤勢，請不要引用你記憶中 2023 或 2024 年的舊聞。
        2. 請根據「技術指標」進行邏輯推演，而不是瞎掰新聞。
        
        【即時數據 ({current_time})】
        * 股價: {data['price']} (漲跌幅: {data['pct']}%)
        * 趨勢: {data['tech']['Trend']}
        * RSI(14): {data['tech']['RSI']} -> 屬於 {rsi_desc}
        * 季線乖離率: {data['tech']['Bias_60']}% (正值代表股價高於季線)
        * 布林通道狀態: {data['tech']['BB_Status']}
        * 成交量狀態: {data['tech']['Vol_Status']}
        
        請用 Markdown 格式撰寫。不要包含 ```markdown 字樣。
        
        架構要求：
        # 【訂戶專屬】{data['name_zh']} ({data['symbol']}) 趨勢診斷 ({current_time})
        
        ## 1. 核心論點 (Executive Summary)
        (請根據漲跌幅與趨勢，給出 3 點結論)
        
        ## 2. 技術面與籌碼解析 (Technical Deep Dive)
        * **趨勢判斷**：目前呈現「{data['tech']['Trend']}」。(請解釋這代表什麼)
        * **RSI 與乖離**：RSI 來到 {data['tech']['RSI']}，顯示... (請解釋是否過熱或背離)
        * **布林與量能**：股價{data['tech']['BB_Status']}，配合{data['tech']['Vol_Status']}，這意味著...
        
        ## 3. 風險與策略 (Risk & Strategy)
        (請給出具體的支撐壓力觀察點，不要給模糊的建議)
        """

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.5
        )
        content = completion.choices[0].message.content
        return content.replace('```markdown', '').replace('```', '').strip()

    except Exception as e:
        print(f"AI 生成錯誤: {e}")
        return f"# 生成失敗\n\nAI 連線錯誤：{str(e)}"

# --- 路由 ---
@app.route('/')
def home(): return render_template('report.html')

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    try:
        user_input = request.json.get('ticker', '').strip().upper()
        
        if user_input == "WEEKLY":
            data = get_macro_data()
            if not data: return jsonify({'error': '宏觀數據獲取失敗，請檢查網路'})
            
            raw_text = generate_report_content("WEEKLY", data)
            return jsonify({'content': markdown.markdown(raw_text), 'raw': raw_text, 'type': 'weekly'})
        else:
            symbol = f"{user_input}.TW" if user_input.isdigit() else user_input
            
            stock = yf.Ticker(symbol)
            try: info = stock.info
            except: info = {}

            tech = get_stock_data_full(symbol)
            if not tech: return jsonify({'error': f'無法取得 {symbol} 數據，請確認代號正確'})
            
            raw_name = info.get('longName', user_input)
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
        print(f"Server Error: {e}")
        return jsonify({'error': str(e)})

@app.route('/explain_simple', methods=['POST'])
def explain_simple():
    raw_text = request.json.get('text', '')
    if not raw_text: return jsonify({'explanation': '無內容'})
    prompt = f"請用非常口語、簡短、幽默的方式總結這段股市分析(像是老手對新手說話)：\n{raw_text[:1500]}"
    try:
        res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":prompt}])
        return jsonify({'explanation': res.choices[0].message.content})
    except: return jsonify({'explanation': '翻譯失敗'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)