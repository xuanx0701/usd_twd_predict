import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime
import pandas as pd
import numpy as np

def fetch_data(start_date="2010-01-01", end_date="2026-05-21"):
    """下載 USD/TWD 歷史資料"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"📥 下載 USD/TWD 資料 ({start_date} ~ {end_date})...")

    ticker = yf.Ticker("TWD=X")
    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError("資料為空")
    
    # 1. 篩選欄位並統一命名為 USDTWD (對齊後面的技術指標函式)
    df = df[['Close']].rename(columns={'Close': 'USDTWD'})
    
    # 2. 處理時間格式並移除時區
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # 3. 排序並過濾掉異常值
    df = df.sort_index()
    df = df[df['USDTWD'] > 0]
    df = df.reset_index()
    df = df.rename(columns={'Date': 'Date', 'index': 'Date'}) # 確保欄位叫 Date

    print(f"✅ 成功下載 {len(df)} 筆資料")
    print(f"   最新匯率: {df['USDTWD'].iloc[-1]:.4f}")
    
    return df

def add_real_data(name: str, dfname: str, df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """
    整合版：支援多種日期格式自動解析、欄位空格清理、並使用 merge_asof 高效對齊
    """
    try:
        # 1. 讀取 CSV (支援多種編碼)
        for enc in ['utf-8-sig', 'big5', 'cp950']:
            try:
                rate_df = pd.read_csv(csv_path, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # 2. 清理欄位名稱 (移除空格、處理中文標頭)
        rate_df.columns = rate_df.columns.str.strip()
        if '日期' in rate_df.columns:
            rate_df = rate_df.rename(columns={'日期': 'Date'})
        
        # 3. 自動解析日期格式 (處理 20240322 或 2024-03-22)
        # 先嘗試 20240322 這種格式，失敗了再用預設解析
        temp_date = pd.to_datetime(rate_df['Date'], format='%Y%m%d', errors='coerce')
        if temp_date.isna().all():
            rate_df['Date'] = pd.to_datetime(rate_df['Date'], errors='coerce')
        else:
            rate_df['Date'] = temp_date
            
        rate_df = rate_df.dropna(subset=['Date']).sort_values('Date')
        
        # 4. 確保數值正確 (避免利率變成字串)
        rate_df[name] = pd.to_numeric(rate_df[name], errors='coerce')

        # 5. 確保主表 Date 格式一致 (Naive datetime)
        if 'Date' not in df.columns:
            df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.sort_values('Date')

        # 6. 檢查目標欄位是否存在
        if name not in rate_df.columns:
            print(f"❌ 警告：在 {csv_path} 中找不到欄位 '{name}'")
            print(f"👉 該 CSV 現有欄位為: {list(rate_df.columns)}")
            return df
        
        # 7. 使用 merge_asof 進行「前值填充合併」
        # 這會自動處理「先填充完每一天再合併」的邏輯，且不佔空間
        df = pd.merge_asof(df, rate_df[['Date', name]], 
                           on='Date', 
                           direction='backward')
        
        # 8. 改名與最終填補 (處理主表日期超出 CSV 範圍的情況)
        df = df.rename(columns={name: dfname})
        df[dfname] = df[dfname].ffill().bfill() 
        
        print(f"✅ 成功整合 {dfname} (來源: {name})")
        
    except Exception as e:
        print(f"❌ 執行 add_real_data 時出錯: {e}")
        
    return df

def load_data():
    df = fetch_data("2010-01-01")
    return df

# ─────────────────────────────────────────────
#  經濟基本面特徵（模擬＋說明）
# ─────────────────────────────────────────────
 
def add_economic_features(df: pd.DataFrame) -> pd.DataFrame:

    if 'Date' not in df.columns:
        df = df.reset_index()

    df['Date'] = pd.to_datetime(df['Date'])
    start_date = df['Date'].min()
    end_date = df['Date'].max()
 
    # ── 聯準會政策利率
    # DFF: 聯邦基金有效利率, VIXCLS: VIX 指數
    fred_data = web.DataReader(['DFF', 'VIXCLS'], 'fred', start_date, end_date)
 
    # ── 台灣重貼現率
    df = add_real_data("重貼現率","tw_rate",df,"EG28D01.csv")

    df = add_real_data("外資買賣超","foreign_net",df,"20260511183052DataExport.csv")
 
    # 金融市場數據 (來源: Yahoo Finance)
    # DX-Y.NYB: 美元指數, ^TWII: 台股指數 (用來算外資動向)
    tickers = yf.download(['DX-Y.NYB', '^TWII'], start=start_date, end=end_date)
    market_data = tickers['Close']

    # 整合 df 
    df = df.merge(fred_data, left_on='Date', right_index=True, how='left')
    df = df.merge(market_data, left_on='Date', right_index=True, how='left')

    # 欄位轉換 
    df.rename(columns={'DFF': 'fed_rate'}, inplace=True)
    df.rename(columns={'DX-Y.NYB': 'dxy'}, inplace=True)

    # ── 利差（美-台）→ 正值代表美元利率更高
    df['rate_diff'] = df['fed_rate'] - df['tw_rate']
     
    # ── 台灣出口景氣（半年週期）
    df = add_real_data("出口總值","export",df,"i9101_2118211305.csv")

    return df


if __name__ == "__main__":
    try:
        # 流程自動化
        processed_df = fetch_data("2010-01-01")
        processed_df = add_economic_features(processed_df)
        
        # 檢查是否有資料
        if not processed_df.empty:
            before_count = len(processed_df)
            processed_df = processed_df.dropna()
            after_count = len(processed_df)
            # ✅ 修正點：直接使用 datetime.now()
            current_time = datetime.now().strftime('%Y%m%d')
            file_name = f"USDTWD_Analysis_{current_time}.csv"
            
            processed_df.to_csv(file_name, index=False, encoding='utf-8-sig')
            print(f"✅ 任務完成！檔案已儲存為：{file_name}")
            print(processed_df.tail()) 
    except Exception as e:
        print(f"❌ 執行過程中出錯: {e}")
