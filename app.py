# app.py
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
import markdown
import numpy as np
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import time

load_dotenv()

# --- 雲端環境修正 ---
try:
    if not os.path.exists('/tmp/yf_cache'):
        os.makedirs('/tmp/yf_cache')
    yf.set_tz_cache_location('/tmp/yf_cache')
except:
    pass

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)

# --- API Keys ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
client_r1 = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
MODEL_R1 = "deepseek-ai/deepseek-r1"

XAI_API_KEY = os.getenv("XAI_API_KEY")
client_xai = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)
MODEL_XAI = "grok-4-1-fast-reasoning"

STOCK_NAME_MAP = {
    '2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '0050.TW': '台灣50',
    '2603.TW': '長榮', 'NVDA': '輝達', 'TSLA': '特斯拉', 'AAPL': '蘋果',
    'BTC-USD': '比特幣', 'ETH-USD': '以太幣', 'SOL-USD': 'Solana',
    'ADA-USD': 'Cardano', 'BNB-USD': '幣安幣', 'GC=F': '黃金'
}

# --- 簡易快取 ---
_cache = {}
def get_cache(key, ttl):
    v = _cache.get(key)
    if not v:
        return None
    data, ts = v
    if time.time() - ts > ttl:
        return None
    return data

def set_cache(key, data):
    _cache[key] = (data, time.time())

def safe_float(val):
    try:
        if pd.isna(val) or val is None:
            return 0.0
        return float(val)
    except:
        return 0.0

def normalize_yf_df(df):
    """
    Normalize yfinance DataFrame to ensure 'Close' and 'Volume' columns exist.
    """
    if df is None or df.empty:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df = df['Close']
            else:
                try:
                    df = df.xs('Close', axis=1, level=0, drop_level=False)
                except:
                    df.columns = [c[1] if isinstance(c, tuple) and len(c) > 1 else c for c in df.columns]
    except:
        pass

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in c if x]) for c in df.columns]

    if 'Close' not in df.columns:
        if len(df.columns) == 1:
            col = df.columns[0]
            df.rename(columns={col: 'Close'}, inplace=True)
        else:
            for cand in ['Adj Close', 'close', 'Close*']:
                if cand in df.columns:
                    df.rename(columns={cand: 'Close'}, inplace=True)
                    break
            if 'Close' not in df.columns:
                return None

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    return df

# --- 右側大盤 ---
def get_market_indices_data():
    c = get_cache("indices", ttl=60)
    if c is not None:
        return c
    try:
        symbols = ['^TWII', '^DJI', '^GSPC', '^NDX']
        df = yf.download(symbols, period="5d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            set_cache("indices", [])
            return []

        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                close_df = df['Close']
                if isinstance(close_df.columns, pd.MultiIndex):
                    close_df.columns = [c[1] for c in close_df.columns]
            else:
                close_df = df
        else:
            close_df = df

        indices = []
        names = {'^TWII': '台股加權', '^DJI': '道瓊工業', '^GSPC': 'S&P 500', '^NDX': 'Nasdaq 100'}
        for sym in symbols:
            try:
                series = close_df[sym].dropna()
                if len(series) >= 2:
                    curr, prev = float(series.iloc[-1]), float(series.iloc[-2])
                    change = (curr - prev) / prev * 100 if prev != 0 else 0
                    indices.append({'name': names.get(sym, sym), 'price': f"{curr:,.0f}", 'change': round(change, 2), 'symbol': sym})
            except:
                continue
        set_cache("indices", indices)
        return indices
    except:
        set_cache("indices", [])
        return []

# --- 宏觀數據 ---
def get_macro_data():
    c = get_cache("macro", ttl=120)
    if c is not None:
        return c
    try:
        assets = {'TWII': '^TWII', 'SPY': 'SPY', 'QQQ': 'QQQ', 'SOX': '^SOX', 'VIX': '^VIX', 'US10Y': '^TNX', 'USD/TWD': 'TWD=X'}
        raw = yf.download(list(assets.values()), period='60d', interval='1d', progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None

        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if 'Close' in df.columns.get_level_values(0):
                    df = df['Close']
            except:
                pass
            new_cols = []
            rev_map = {v: k for k, v in assets.items()}
            for col in df.columns:
                k = col[1] if isinstance(col, tuple) else col
                new_cols.append(rev_map.get(k, k))
            df.columns = new_cols

        prices, weekly_change = {}, {}
        for col in df.columns:
            if len(df) >= 5:
                prev, curr = safe_float(df[col].iloc[-5]), safe_float(df[col].iloc[-1])
                change = (curr - prev) / prev if prev != 0 else 0
                weekly_change[col] = f"{'+' if change>0 else ''}{round(change*100, 2)}%"
                prices[col] = round(curr, 2)

        data = {"weekly_change": weekly_change, "prices": prices}
        set_cache("macro", data)
        return data
    except:
        return None

# --- 個股數據 ---
def get_stock_data_full(symbol):
    try:
        raw = yf.download(symbol, period="180d", interval="1d", progress=False, auto_adjust=True)
        df = normalize_yf_df(raw)
        if df is None or df.empty:
            return None

        # Technicals
        try:
            df['RSI'] = ta.rsi(df['Close'], length=14)
        except:
            df['RSI'] = 50.0

        try:
            df['SMA_60'] = ta.sma(df['Close'], length=60)
        except:
            df['SMA_60'] = df['Close'].rolling(60).mean()

        # MACD
        try:
            macd = ta.macd(df['Close'])
            macd_cols = macd.columns.tolist() if macd is not None and hasattr(macd, 'columns') else []
            df['MACD'] = macd['MACD_12_26_9'] if 'MACD_12_26_9' in macd_cols else (macd.iloc[:,0] if macd is not None and macd.shape[1] > 0 else 0.0)
            df['MACD_Hist'] = macd['MACDh_12_26_9'] if 'MACDh_12_26_9' in macd_cols else (macd.iloc[:,1] if macd is not None and macd.shape[1] > 1 else 0.0)
            df['MACD_Signal'] = macd['MACDs_12_26_9'] if 'MACDs_12_26_9' in macd_cols else (macd.iloc[:,2] if macd is not None and macd.shape[1] > 2 else 0.0)
        except:
            df['MACD'] = 0.0
            df['MACD_Hist'] = 0.0
            df['MACD_Signal'] = 0.0

        try:
            std = df['Close'].rolling(20).std()
            sma20 = df['Close'].rolling(20).mean()
            df['BB_Upper'] = sma20 + 2 * std
        except:
            df['BB_Upper'] = df['Close'] * 1.02

        try:
            df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        except:
            df['Vol_SMA'] = df['Volume']

        df = df.fillna(method='ffill').fillna(0)
        if len(df) < 2:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        change_val = safe_float(last['Close'] - prev['Close'])
        change_pct = safe_float((change_val / prev['Close']) * 100) if prev['Close'] != 0 else 0

        # Fundamental
        try:
            info = yf.Ticker(symbol).info or {}
            fund = {"PE": info.get('trailingPE', 'N/A'), "EPS": info.get('trailingEps', 'N/A'), "PB": info.get('priceToBook', 'N/A')}
        except:
            fund = {"PE": "N/A", "EPS": "N/A", "PB": "N/A"}

        sma_60 = safe_float(last.get('SMA_60', 0))
        close = safe_float(last['Close'])
        trend_str = "多頭" if (sma_60 != 0 and close > sma_60) else "空頭"

        chart_len = min(60, len(df))
        chart_df = df.iloc[-chart_len:]

        return {
            "price": round(close, 2),
            "change": round(change_val, 2),
            "pct": round(change_pct, 2),
            "Fundamental": fund,
            "tech": {
                "RSI": round(safe_float(last.get('RSI', 50)), 2),
                "Trend": trend_str,
                "Vol_Status": "放量" if last['Volume'] > safe_float(last.get('Vol_SMA', 0)) * 1.2 else "縮量",
                "BB_Status": "觸及上軌" if close >= safe_float(last.get('BB_Upper', 1e9)) else "正常",
                "Bias_60": round(((close - sma_60) / sma_60) * 100, 2) if sma_60 != 0 else 0
            },
            "chart": {
                "dates": chart_df.index.strftime('%m/%d').tolist(),
                "prices": [safe_float(x) for x in chart_df['Close'].tolist()],
                "volumes": [safe_float(x) for x in chart_df['Volume'].tolist()],
                "macd": [safe_float(x) for x in chart_df['MACD'].tolist()],
                "macd_hist": [safe_float(x) for x in chart_df['MACD_Hist'].tolist()],
                "macd_signal": [safe_float(x) for x in chart_df['MACD_Signal'].tolist()]
            }
        }
    except Exception as e:
        print(f"Stock Data Error: {e}")
        return None

# --- AI 呼叫 ---
def call_ai(client, model, sys, user, timeout_sec=40, retries=1):
    for attempt in range(retries + 1):
        try:
            if not client.api_key:
                return "Error: API Key Missing"
            start = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.7
            )
            if time.time() - start > timeout_sec:
                raise TimeoutError("LLM response timeout")
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries:
                time.sleep(0.5)
                continue
            return f"Error: {str(e)}"

def generate_dual_model_report(input_type, data):
    current_time = datetime.now().strftime("%Y-%m-%d")

    if input_type == "WEEKLY":
        context = f"""
        時間: {current_time}
        數據: 台股{data['weekly_change'].get('TWII')}, 美股{data['weekly_change'].get('SPY')}, VIX {data['prices'].get('VIX')}
        """
        task_specific = "重點分析美債殖利率對科技股影響。"
    else:
        fund = data['Fundamental']
        context = f"""
        標的: {data['name_zh']} ({data['symbol']}) | 價格: {data['price']} | 漲跌: {data['pct']}%
        技術面: {data['tech']['Trend']} | RSI: {data['tech']['RSI']} | 量能: {data['tech']['Vol_Status']}
        基本面: PE {fund['PE']} | EPS {fund['EPS']}
        """
        task_specific = "分析主力籌碼、技術乖離與估值。"

    r1_sys = "你是一名資深金融分析師。"
    r1_user = f"【資料】{context}\n【任務】{task_specific}\n【框架】1.宏觀背景 2.數據解讀 3.情境推演(基準/樂觀/悲觀) 4.結論。拒絕臆測。"
    r1_res = call_ai(client_r1, MODEL_R1, r1_sys, r1_user, timeout_sec=40, retries=1)
    if isinstance(r1_res, str) and r1_res.startswith("Error"):
        r1_res = ("深度模型目前不可用，改用基礎框架生成摘要：\n"
                  "1) 宏觀背景：近期待觀望。\n"
                  "2) 數據解讀：波動加劇，留意殖利率變化。\n"
                  "3) 情境：基準-持平；樂觀-科技反彈；悲觀-風險資產回檔。\n"
                  "4) 結論：分批、設停損。")

    xai_sys = "你是擅長引導新手的投資顧問。"
    xai_user = f"""
    任務：將報告轉為新手建議。
    【深度分析】{r1_res}
    ⚠️ 【輸出格式 (Markdown)】
    ### 🛣️ 市場紅綠燈
    * **燈號**：(🟢/🟡/🔴)
    * **一句話建議**：(引用區塊 >)
    ### 🎯 策略與機會
    | 投資風格 | 建議策略 | 操作指引 |
    | :--- | :--- | :--- |
    | 🛡️ 保守 | ... | ... |
    | ⚖️ 穩健 | ... | ... |
    | 🚀 積極 | ... | ... |
    ### 📝 深度解析
    1. 核心觀點
    2. 關鍵風險
    """
    xai_res = call_ai(client_xai, MODEL_XAI, xai_sys, xai_user, timeout_sec=40, retries=1)
    if isinstance(xai_res, str) and xai_res.startswith("Error"):
        xai_res = """
### 🛣️ 市場紅綠燈
* **燈號**：🟡
* **一句話建議**：
> 市場變動加劇，分批進場、嚴守停損。

### 🎯 策略與機會
| 投資風格 | 建議策略 | 操作指引 |
| :--- | :--- | :--- |
| 🛡️ 保守 | 持有現金為主，逢低小量布局 | 先觀察 3-5 個交易日，跌破季線不加碼 |
| ⚖️ 穩健 | 核心部位不動，衛星部位做波段 | 設 5-8% 停損，量縮不追價 |
| 🚀 積極 | 聚焦龍頭與高景氣族群 | 追多日均線上彎、量價齊升標的 |

### 📝 深度解析
1. 核心觀點：殖利率變動牽動估值，科技股波動放大。
2. 關鍵風險：財報/指引不如預期、政策與地緣風險、流動性收縮。
        """.strip()

    return xai_res.replace('```markdown', '').replace('```', '').strip()

# --- Routes ---
@app.route('/')
def home():
    return render_template('report.html')

@app.route('/get_indices', methods=['GET'])
def get_indices():
    return jsonify(get_market_indices_data())

@app.route('/get_news', methods=['POST'])
def get_stock_news():
    try:
        user_input = request.json.get('ticker', '').strip().upper()
        symbol = f"{user_input}.TW" if user_input.isdigit() else user_input
        if user_input in ['BTC', 'ETH', 'SOL', 'ADA', 'BNB']:
            symbol = f"{user_input}-USD"

        cache_key = f"news:{symbol}"
        c = get_cache(cache_key, ttl=120)
        if c is not None:
            return jsonify({'news': c, 'source': 'cache'})

        t = yf.Ticker(symbol)
        raw = None
        try:
            raw = t.news
        except Exception as e:
            print(f"[NEWS] yfinance error for {symbol}: {e}")

        if not raw:
            # 明確回傳原因，方便前端提示
            return jsonify({'news': [], 'reason': 'no_source', 'symbol': symbol}), 200

        clean = []
        for n in raw:
            try:
                title = n.get('title')
                link = n.get('link')
                if not title or not link:
                    continue
                ts = n.get('providerPublishTime', 0)
                try:
                    pt = datetime.fromtimestamp(ts).strftime('%m/%d %H:%M') if ts else "Recent"
                except:
                    pt = "Recent"
                clean.append({
                    'title': title,
                    'link': link,
                    'publisher': n.get('publisher', 'News'),
                    'time': pt
                })
            except Exception as e:
                print(f"[NEWS] parse one item failed: {e}")
                continue

        clean = clean[:6]
        set_cache(cache_key, clean)
        return jsonify({'news': clean, 'source': 'yfinance'})
    except Exception as e:
        print(f"[NEWS] endpoint error: {e}")
        return jsonify({'news': [], 'reason': 'server_error'}), 200

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    try:
        user_input = request.json.get('ticker', '').strip().upper()
        if user_input == "WEEKLY":
            data = get_macro_data()
            if not data:
                return jsonify({'error': 'Error fetching macro data'})
            report = generate_dual_model_report("WEEKLY", data)
            return jsonify({
                'content': markdown.markdown(report),
                'raw': report,
                'type': 'weekly',
                'model_info': 'DeepSeek R1 (Reasoning) + Grok (Guidance)'
            })
        else:
            symbol = f"{user_input}.TW" if user_input.isdigit() else user_input
            if user_input in ['BTC', 'ETH', 'SOL', 'ADA', 'BNB']:
                symbol = f"{user_input}-USD"
            tech = get_stock_data_full(symbol)
            if not tech:
                return jsonify({'error': 'Invalid Ticker or no data available'})
            stock_data = {'symbol': user_input, 'name_zh': STOCK_NAME_MAP.get(symbol, user_input), **tech}
            report = generate_dual_model_report("STOCK", stock_data)
            return jsonify({
                'content': markdown.markdown(report),
                'raw': report,
                'meta': stock_data,
                'type': 'stock',
                'chartData': tech['chart'],
                'model_info': 'DeepSeek R1 (Reasoning) + Grok (Guidance)'
            })
    except Exception as e:
        return jsonify({'error': str(e)})

# --- 新增：雙模型投顧建議端點 ---
@app.route('/advise', methods=['POST'])
def advise():
    try:
        payload = request.json or {}
        meta = payload.get('meta')
        raw = payload.get('raw', '')
        if not meta or not isinstance(meta, dict):
            return jsonify({'error': 'no_meta'}), 400

        base_ctx = f"""
        標的: {meta.get('name_zh', meta.get('symbol'))} ({meta.get('symbol')})
        現價: {meta.get('price')}  漲跌: {meta.get('pct')}%
        技術: 趨勢 {meta.get('tech',{}).get('Trend')} / RSI {meta.get('tech',{}).get('RSI')} / 量能 {meta.get('tech',{}).get('Vol_Status')}
        其他: 60日乖離 {meta.get('tech',{}).get('Bias_60')}%
        """

        # 第一階段：R1 專業條列
        r1_sys = "你是一名嚴謹的資深金融分析師。"
        r1_user = f"請根據以下資訊產出三點最重要的操作建議（條列、避免贅詞）：\n{base_ctx}\n\n參考報告：\n{raw}\n\n格式：\n1) 進出場條件\n2) 風險控管\n3) 候選觀察清單(若有)"
        r1_res = call_ai(client_r1, MODEL_R1, r1_sys, r1_user, timeout_sec=40, retries=1)
        if isinstance(r1_res, str) and r1_res.startswith("Error"):
            r1_res = "1) 進出場：均線上彎、量價齊升時分批；破季線停損。\n2) 風險控管：5-8% 停損；財報/指引異常時降風險。\n3) 觀察清單：同族群龍頭與高景氣標的。"

        # 第二階段：Grok 新手可執行步驟
        xai_sys = "你是擅長把專業決策翻譯成白話的投顧老師。"
        xai_user = f"把以下分析壓縮成【100字內】新手可執行的步驟，列點、直接可操作：\n{r1_res}"
        xai_res = call_ai(client_xai, MODEL_XAI, xai_sys, xai_user, timeout_sec=40, retries=1)
        if isinstance(xai_res, str) and xai_res.startswith("Error"):
            xai_res = "步驟：1. 等均線上彎且放量時小量買進；2. 分批加碼，每次不超過總資金 20%；3. 跌破季線或虧損 7% 立即停損。"

        return jsonify({'advice_raw': r1_res.strip(), 'advice_simple': xai_res.strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/healthz')
def healthz():
    return jsonify({"status": "ok", "time": datetime.now().isoformat(), "version": "v1"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
