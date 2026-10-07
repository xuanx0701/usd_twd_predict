"""
=============================================================
  USD/TWD 匯率機器學習預測系統
  目標：預測 2025/5/29 台幣兌美元匯率
  作業截止日：2025/5/20
=============================================================
  方法架構：
  1. 經濟基本面特徵（利差、通膨、貿易、DXY）
  2. 技術分析指標（MA, RSI, MACD, Bollinger Bands）
  3. 集成模型（Random Forest + Gradient Boosting + Ridge 加權）
  4. 時間序列交叉驗證
=============================================================
"""

import numpy as np
import pandas as pd
import warnings
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from matplotlib import rcParams
import yfinance as yf

import pandas_datareader.data as web

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  字體設定（支援中文）
# ─────────────────────────────────────────────
rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
rcParams['axes.unicode_minus'] = False

print("=" * 65)
print("   USD/TWD 匯率機器學習預測系統")
print("   目標預測日：2026/05/29")
print("=" * 65)


# ─────────────────────────────────────────────
#  資料獲取模組
# ─────────────────────────────────────────────
def fetch_data(start_date="2010-01-01", end_date="2026-05-20"):
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
#  技術分析指標計算
# ─────────────────────────────────────────────

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算常見技術分析指標"""
    p = df['USDTWD'].copy()
 
    # 移動平均
    for w in [5, 10, 20, 60]:
        df[f'MA{w}'] = p.rolling(w).mean()
        df[f'MA{w}_slope'] = df[f'MA{w}'].diff(5) / 5
 
    # 指數移動平均
    df['EMA12'] = p.ewm(span=12, adjust=False).mean()
    df['EMA26'] = p.ewm(span=26, adjust=False).mean()
 
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
 
    # RSI（14日）
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))
 
    # Bollinger Bands（20日，2標準差）
    df['BB_mid'] = p.rolling(20).mean()
    df['BB_std'] = p.rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
    df['BB_pct'] = (p - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-8)
 
    # 報酬率與波動度
    for lag in [1, 3, 5, 10, 20]:
        df[f'ret_{lag}d'] = p.pct_change(lag)
    df['volatility_10d'] = p.pct_change().rolling(10).std() * np.sqrt(252)
    df['volatility_20d'] = p.pct_change().rolling(20).std() * np.sqrt(252)
 
    # 動能指標
    df['momentum_5'] = p - p.shift(5)
    df['momentum_20'] = p - p.shift(20)
 
    # 乖離率（與MA20之距離）
    df['bias20'] = (p - df['MA20']) / df['MA20']
 
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
 
    # 金融市場數據 (來源: Yahoo Finance)
    # DX-Y.NYB: 美元指數, ^TWII: 台股指數 (用來算外資動向代理)
    tickers = yf.download(['DX-Y.NYB', '^TWII'], start=start_date, end=end_date)
    market_data = tickers['Close']

    # 整合 df 
    df = df.merge(fred_data, left_on='Date', right_index=True, how='left')
    df = df.merge(market_data, left_on='Date', right_index=True, how='left')

    # 欄位轉換 
    df['DX-Y.NYB'] = df['DX-Y.NYB'].ffill().bfill()
    df['^TWII'] = df['^TWII'].ffill().bfill()
    df['vix_lag3'] = df['VIXCLS'].shift(3)
    df['dxy_ma20'] = df['DX-Y.NYB'].rolling(20).mean()
    df['dxy_momentum'] = df['DX-Y.NYB'].diff(10)

    # ── 利差（美-台）→ 正值代表美元利率更高
    df['rate_diff'] = df['DFF'] - df['tw_rate']
    df['rate_diff_lag5'] = df['rate_diff'].shift(5)
     
    # ── 台灣出口景氣代理（半年週期）
    df = add_real_data("出口總值","export",df,"i9101_2118211305.csv")
    df['export_lag20'] = df['export'].shift(20).values  # 貿易數據延遲

    df = add_real_data("外資買賣超","foreign_net",df,"20260511183052DataExport.csv")

    return df
 
 
# ─────────────────────────────────────────────
#  特徵工程彙整
# ─────────────────────────────────────────────
 
def build_features(df: pd.DataFrame, target_horizon: int = 5) -> tuple:
    """
    組合所有特徵，建立監督式學習資料集
    target_horizon: 預測幾個交易日後的匯率
    """
    df = compute_technical_indicators(df)
    df = add_economic_features(df)
 
    # 目標變數：N個交易日後的收盤價
    df['target'] = df['USDTWD'].shift(-target_horizon)
 
    feature_cols = [
        # ── 技術面
        'MA5', 'MA10', 'MA20', 'MA60',
        'MA5_slope', 'MA20_slope',
        'EMA12', 'EMA26',
        'MACD', 'MACD_signal', 'MACD_hist',
        'RSI14',
        'BB_width', 'BB_pct',
        'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
        'volatility_10d', 'volatility_20d',
        'momentum_5', 'momentum_20',
        'bias20',
 
        # ── 經濟基本面
        'DFF', 'tw_rate', 'rate_diff', 'rate_diff_lag5',
        'DX-Y.NYB', 'dxy_ma20', 'dxy_momentum',
        'VIXCLS', 'vix_lag3',
        'export', 'export_lag20','foreign_net'
    ]

    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
 
    X = df_clean[feature_cols].values
    y = df_clean['target'].values
    dates_clean = df_clean['Date'].values
 
    return X, y, dates_clean, feature_cols, df_clean
 
# ─────────────────────────────────────────────
#  集成模型訓練
# ─────────────────────────────────────────────
 
class EnsemblePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42
            ),
            'Ridge Regression': Ridge(alpha=10.0),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        }
        self.weights = None
        self.val_scores = {}
 
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_scaled = self.scaler.fit_transform(X_train)
 
        val_maes = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                pred = model.predict(X_val_scaled)
                mae = mean_absolute_error(y_val, pred)
                val_maes[name] = mae
                self.val_scores[name] = mae
 
        # 依驗證MAE計算加權（MAE越低，權重越高）
        if val_maes:
            inv = {k: 1.0 / v for k, v in val_maes.items()}
            total = sum(inv.values())
            self.weights = {k: v / total for k, v in inv.items()}
        else:
            self.weights = {k: 0.25 for k in self.models}
 
        return self
 
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            p = model.predict(X_scaled)
            preds += self.weights[name] * p
        return preds
 
    def predict_individual(self, X):
        X_scaled = self.scaler.transform(X)
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict(X_scaled)
        return results
 
    def get_feature_importance(self, feature_names):
        """獲取 Random Forest 特徵重要性"""
        rf = self.models['Random Forest']
        importances = rf.feature_importances_
        fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
        return fi.sort_values('importance', ascending=False)


    def get_shap_values(self, X, feature_names):
        """計算 Random Forest 的 SHAP values。"""
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        rf = self.models['Random Forest']
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer(X_scaled_df)

        return explainer, shap_values, X_scaled_df
 
 
# ─────────────────────────────────────────────
#  時間序列交叉驗證
# ─────────────────────────────────────────────
 
def time_series_cv(X, y, dates):
    """5折時間序列交叉驗證"""
    print("\n[3/4] 時間序列交叉驗證...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []
 
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
 
        predictor = EnsemblePredictor()
        predictor.fit(X_tr, y_tr, X_va, y_va)
        y_pred = predictor.predict(X_va)
 
        mae = mean_absolute_error(y_va, y_pred)
        rmse = np.sqrt(mean_squared_error(y_va, y_pred))
        r2 = r2_score(y_va, y_pred)
        mape = np.mean(np.abs((y_va - y_pred) / y_va)) * 100
 
        cv_results.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE(%)': mape
        })
        print(f"    Fold {fold+1}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")
 
    cv_df = pd.DataFrame(cv_results)
    print(f"\n    平均 MAE  = {cv_df['MAE'].mean():.4f} ± {cv_df['MAE'].std():.4f}")
    print(f"    平均 RMSE = {cv_df['RMSE'].mean():.4f} ± {cv_df['RMSE'].std():.4f}")
    print(f"    平均 MAPE = {cv_df['MAPE(%)'].mean():.2f}%")
    return cv_df
 
 
# ─────────────────────────────────────────────
#  繪圖
# ─────────────────────────────────────────────
 
def plot_results(df_full, y_test, y_pred_test, dates_test,
                 cv_df, feature_importance, individual_preds,
                 final_prediction, prediction_date, output_dir):
    """產出圖表（含 Learning Curve）"""
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#0d1117')
 
    C = {
        'bg':    "#ffffff",
        'grid':  "#d8d8d8",
        'text':  "#000000",
        'text2': "#555b61",
        'blue':  '#58a6ff',
        'red':   '#f78166',
        'green': '#3fb950',
        'amber': '#ffa657',
        'purple':'#d2a8ff',
    }
 
    def style_ax(ax, title, xlabel=None, ylabel=None):
        ax.set_facecolor(C['bg'])
        ax.set_title(title, color=C['text'], pad=10, fontsize=11)
        if xlabel: ax.set_xlabel(xlabel, color=C['text2'])
        if ylabel: ax.set_ylabel(ylabel, color=C['text2'])
        ax.tick_params(colors=C['text2'], labelsize=8)
        ax.grid(True, color=C['grid'], linewidth=0.5)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']: ax.spines[sp].set_color(C['grid'])
 
    def legend(ax):
        ax.legend(facecolor=C['bg'], edgecolor=C['grid'],
                  labelcolor=C['text'], fontsize=8)
 
    # ── 圖1：歷史匯率 + 技術指標
    ax1 = fig.add_subplot(3, 3, (1, 3))
    recent = df_full.tail(500)
    ax1.plot(recent['Date'], recent['USDTWD'], color=C['blue'], linewidth=1.0, label='USD/TWD')
    ax1.plot(recent['Date'], recent['MA20'], color=C['amber'], linewidth=1.4, linestyle='--', alpha=0.85, label='MA20')
    ax1.plot(recent['Date'], recent['MA60'], color=C['purple'], linewidth=1.4, linestyle='--', alpha=0.85, label='MA60')
    ax1.fill_between(recent['Date'], recent['BB_upper'], recent['BB_lower'],
                     color=C['blue'], alpha=0.08, label='Bollinger Band')
    style_ax(ax1, 'USD/TWD Historical Rate + Technical Indicators (Last 500 Days)', ylabel='Rate (TWD/USD)')
    legend(ax1)
 
 
    # ── 圖3：測試集預測 vs 實際
    ax3 = fig.add_subplot(3, 3, 4)
    ax3.plot(dates_test, y_test, color=C['blue'], linewidth=1.5, label='Actual')
    ax3.plot(dates_test, y_pred_test, color=C['red'], linewidth=1.5, linestyle='--', label='Ensemble Pred.')
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred_test))
    ax3.fill_between(dates_test, y_pred_test - rmse_val, y_pred_test + rmse_val,
                     color=C['green'], alpha=0.15, label=f'±1 RMSE ({rmse_val:.3f})')
    style_ax(ax3, 'Test Set: Actual vs Predicted', ylabel='Rate (TWD/USD)')
    legend(ax3)
 
    # ── 圖4：殘差分析（Residual Plot）★ 新增
    ax4 = fig.add_subplot(3, 3, 5)
    residuals = y_test - y_pred_test
    ax4.scatter(y_pred_test, residuals, color=C['blue'], alpha=0.4, s=12, label='Residuals')
    ax4.axhline(0, color=C['red'], linewidth=1.5, linestyle='--')
    ax4.axhline(rmse_val, color=C['amber'], linewidth=1.0, linestyle=':', alpha=0.7, label=f'+RMSE={rmse_val:.3f}')
    ax4.axhline(-rmse_val, color=C['amber'], linewidth=1.0, linestyle=':', alpha=0.7, label=f'-RMSE={rmse_val:.3f}')
    style_ax(ax4, 'Residual Analysis\n(Actual - Predicted)', xlabel='Predicted Value', ylabel='Residual (TWD)')
    legend(ax4)
 
    # ── 圖5：訓練集 vs 測試集 loss bar（各模型）
    ax5 = fig.add_subplot(3, 3, 6)
    model_names_short = list(individual_preds.keys())
    train_maes = []
    test_maes_list = []
 
    # 此處使用驗證 MAE 代表 val loss（已在 EnsemblePredictor.val_scores 記錄）
    model_colors_list = [C['blue'], C['red'], C['purple'], C['amber'], C['green']]
    x = np.arange(len(model_names_short))
    width = 0.35
 
    # 簡單用 val_scores（驗證集MAE）作為「test loss」展示
    # 訓練集 MAE 通常略低（從 overfitting 角度示意）
    val_maes = list(individual_preds.values())  # placeholder
    # 改用 feature_importance placeholder — 實際上畫 val MAE
    ax5.set_facecolor(C['bg'])
 
    # 畫 CV 各 fold 的 train/val loss 趨勢
    folds = cv_df['fold'].values
    ax5.bar(folds - 0.18, cv_df['MAE'], width=0.33, color=C['blue'], alpha=0.85, label='CV Val MAE')
    ax5.bar(folds + 0.18, cv_df['RMSE'], width=0.33, color=C['red'], alpha=0.85, label='CV Val RMSE')
    ax5.axhline(cv_df['MAE'].mean(), color=C['blue'], linestyle='--', alpha=0.6, linewidth=1.2)
    ax5.axhline(cv_df['RMSE'].mean(), color=C['red'], linestyle='--', alpha=0.6, linewidth=1.2)
    style_ax(ax5, 'Time-Series CV: MAE & RMSE per Fold', xlabel='Fold', ylabel='Error (TWD)')
    legend(ax5)
 
    # ── 圖6：特徵重要性
    ax6 = fig.add_subplot(3, 3, 7)
    ax6.set_facecolor(C['bg'])
    top15 = feature_importance.head(15)
    bars = ax6.barh(top15['feature'][::-1], top15['importance'][::-1], color=C['blue'], alpha=0.85)
    tech_kw = ['MA', 'EMA', 'MACD', 'RSI', 'BB', 'ret_', 'vol', 'mom', 'bias']
    econ_kw = ['fed_', 'tw_', 'rate_', 'dxy', 'vix', 'export', 'foreign','DFF','DX-Y.NYB']
    for bar, fname in zip(bars, top15['feature'][::-1]):
        if any(k in fname for k in tech_kw): bar.set_color(C['amber'])
        elif any(k in fname for k in econ_kw): bar.set_color(C['green'])
    style_ax(ax6, 'Feature Importance (Random Forest)')
    ax6.tick_params(axis='y', labelsize=7)
    ax6.legend(handles=[
        mpatches.Patch(color=C['amber'], label='Technical'),
        mpatches.Patch(color=C['green'], label='Economic'),
    ], facecolor=C['bg'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=7)
 
    # ── 圖7：Train vs Test 分割線圖
    ax7 = fig.add_subplot(3, 3, 8)
    ax7.set_facecolor(C['bg'])
    all_dates = df_full['Date'].values
    all_prices = df_full['USDTWD'].values
    split_date = dates_test[0]
    train_mask = all_dates < split_date
    test_mask  = all_dates >= split_date
    ax7.plot(all_dates[train_mask], all_prices[train_mask],
             color=C['blue'], linewidth=1.0, label='Training set', alpha=0.9)
    ax7.plot(all_dates[test_mask], all_prices[test_mask],
             color=C['amber'], linewidth=1.0, label='Test set', alpha=0.9)
    ax7.axvline(split_date, color=C['red'], linewidth=2.0, linestyle='--', label='Train/Test split')
    ax7.fill_between(all_dates[train_mask], all_prices[train_mask].min(),
                     all_prices[train_mask].max(), color=C['blue'], alpha=0.05)
    ax7.fill_between(all_dates[test_mask], all_prices[test_mask].min(),
                     all_prices[test_mask].max(), color=C['amber'], alpha=0.05)
    style_ax(ax7, 'Train / Test Split (80% / 20%)', ylabel='Rate (TWD/USD)')
    legend(ax7)
 
    # ── 圖8：最終預測 bar
    ax8 = fig.add_subplot(3, 3, 9)
    ax8.set_facecolor(C['bg'])
    mnames = list(individual_preds.keys()) + ['Ensemble']
    mvals  = [individual_preds[k] for k in individual_preds] + [final_prediction]
    mcols  = [C['blue'], C['red'], C['purple'], C['amber'], C['green']] + [C['green']]
    mcols  = mcols[:len(mnames)]
    bars8  = ax8.bar(range(len(mnames)), mvals, color=mcols, alpha=0.85)
    ax8.axhline(final_prediction, color=C['green'], linestyle='--', linewidth=2,
                label=f'Ensemble: {final_prediction:.4f}')
    for bar, val in zip(bars8, mvals):
        ax8.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', color=C['text'], fontsize=7)
    ax8.set_xticks(range(len(mnames)))
    ax8.set_xticklabels([n.replace(' ', '\n') for n in mnames], fontsize=7)
    style_ax(ax8, f'Final Prediction: {prediction_date}', ylabel='Rate (TWD/USD)')
    legend(ax8)
 
    plt.suptitle('USD/TWD Exchange Rate Prediction System',
                 color=C['text'], fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout(pad=2.5)
    plot_path = os.path.join(output_dir, 'usd_twd_prediction.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor="#ffffff")
    plt.close()
    print(f"\n    圖表已儲存：{plot_path}")
    return plot_path
 
 

# ─────────────────────────────────────────────
#  SHAP 模型解釋
# ─────────────────────────────────────────────

def run_shap_analysis(predictor, X_test, X_latest, feature_cols, output_dir):
    """
    使用 Random Forest 進行 SHAP 分析。
    輸出：
      - shap_beeswarm.png：全域重要性與影響方向
      - shap_bar.png：平均絕對 SHAP value 排名
      - shap_waterfall_latest.png：最新一筆資料的預測解釋
      - shap_importance.csv：平均絕對 SHAP value 排名
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n    === SHAP 分析（Random Forest）===")

    # 取測試集最近最多 500 筆，避免 SHAP 計算過久
    X_shap = X_test[-min(500, len(X_test)):]

    _, shap_values, _ = predictor.get_shap_values(X_shap, feature_cols)

    # 1. Beeswarm：同時看重要性、特徵值高低與預測影響方向
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    beeswarm_path = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()

    # 2. Bar：以 mean(|SHAP value|) 顯示全域特徵重要性
    shap.plots.bar(shap_values, max_display=15, show=False)
    bar_path = os.path.join(output_dir, "shap_bar.png")
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()

    # 3. Waterfall：解釋最新一筆資料在 Random Forest 中的預測形成
    _, latest_shap, _ = predictor.get_shap_values(X_latest, feature_cols)
    shap.plots.waterfall(latest_shap[0], max_display=15, show=False)
    waterfall_path = os.path.join(output_dir, "shap_waterfall_latest.png")
    plt.savefig(waterfall_path, dpi=180, bbox_inches="tight")
    plt.close()

    # 4. 儲存 SHAP 全域重要性
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_csv_path = os.path.join(output_dir, "shap_importance.csv")
    shap_importance.to_csv(shap_csv_path, index=False)

    print(f"    SHAP beeswarm：{beeswarm_path}")
    print(f"    SHAP bar：{bar_path}")
    print(f"    SHAP waterfall：{waterfall_path}")
    print(f"    SHAP importance CSV：{shap_csv_path}")

    return shap_importance


# ─────────────────────────────────────────────
#  主程式
# ─────────────────────────────────────────────

def main():
    OUTPUT_DIR = "./output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TARGET_DATE = "2026-05-29"
    TARGET_HORIZON = 14  # 5/15 → 5/29 約14個交易日
 
    # ── 步驟1：載入資料
    df = load_data()
    print(f"    資料範圍：{df['Date'].min().date()} ~ {df['Date'].max().date()}")
    print(f"    總筆數：{len(df)} 筆")
 
    # ── 步驟2：特徵工程
    print("\n[2/4] 特徵工程與資料前處理...")
    X, y, dates, feature_cols, df_clean = build_features(df, target_horizon=TARGET_HORIZON)
    print(f"    特徵數量：{len(feature_cols)}")
    print(f"    有效樣本：{len(X)}")
 
    # ── 步驟3：交叉驗證
    cv_df = time_series_cv(X, y, dates)
 
    # ── 步驟4：訓練最終模型（80% 訓練 / 20% 測試）
    print("\n[4/4] 訓練最終集成模型並預測...")
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]
 
    predictor = EnsemblePredictor()
    predictor.fit(X_train, y_train, X_test, y_test)
 
    y_pred_test = predictor.predict(X_test)
 
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
 
    print(f"\n    === 測試集評估結果 ===")
    print(f"    MAE  = {mae_test:.4f} 台幣")
    print(f"    RMSE = {rmse_test:.4f} 台幣")
    print(f"    R²   = {r2_test:.4f}")
    print(f"    MAPE = {mape_test:.2f}%")
 
    print(f"\n    === 各模型驗證 MAE ===")
    for name, score in predictor.val_scores.items():
        weight = predictor.weights[name]
        print(f"    {name:22s}  MAE={score:.4f}  Weight={weight:.3f}")
    # ── 最新資料預測 2025/5/29
    X_latest = X[-1:].copy()  # 使用最新一筆資料點
    individual_preds = predictor.predict_individual(X_latest)
    individual_preds_scalar = {k: float(v[0]) for k, v in individual_preds.items()}
    final_pred = float(predictor.predict(X_latest)[0])
 
    print(f"\n{'=' * 65}")
    print(f"  🎯  預測目標日：{TARGET_DATE} (USD/TWD)")
    print(f"{'=' * 65}")
    for name, val in individual_preds_scalar.items():
        print(f"  {name:22s} → {val:.4f}")
    print(f"{'─' * 65}")
    print(f"  集成模型 (加權平均)  → {final_pred:.4f}")
    print(f"  預測信賴區間 (±RMSE) → [{final_pred - rmse_test:.4f}, {final_pred + rmse_test:.4f}]")
    print(f"{'=' * 65}")
 
    # ── 特徵重要性
    feature_importance = predictor.get_feature_importance(feature_cols)


    # ── SHAP 分析（Random Forest）
    shap_importance = run_shap_analysis(
        predictor=predictor,
        X_test=X_test,
        X_latest=X_latest,
        feature_cols=feature_cols,
        output_dir=OUTPUT_DIR,
    )
 
    # ── 繪圖
    plot_results(
        df_full=df_clean,
        y_test=y_test, y_pred_test=y_pred_test, dates_test=dates_test,
        cv_df=cv_df,
        feature_importance=feature_importance,
        individual_preds=individual_preds_scalar,
        final_prediction=final_pred,
        prediction_date=TARGET_DATE,
        output_dir=OUTPUT_DIR,
    )
 
    # ── 儲存結果
    results = {
        "prediction_target_date": TARGET_DATE,
        "model_run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_prediction_USDTWD": round(final_pred, 4),
        "confidence_interval": {
            "lower": round(final_pred - rmse_test, 4),
            "upper": round(final_pred + rmse_test, 4)
        },
        "individual_model_predictions": {k: round(v, 4) for k, v in individual_preds_scalar.items()},
        "model_weights": {k: round(v, 4) for k, v in predictor.weights.items()},
        "test_set_metrics": {
            "MAE": round(mae_test, 4),
            "RMSE": round(rmse_test, 4),
            "R2": round(r2_test, 4),
            "MAPE_%": round(mape_test, 2)
        },
        "cv_avg_MAE": round(cv_df['MAE'].mean(), 4),
        "cv_avg_RMSE": round(cv_df['RMSE'].mean(), 4),
        "top10_features": feature_importance.head(10)['feature'].tolist(),
        "top10_shap_features": shap_importance.head(10)['feature'].tolist(),
        "data_range": {
            "start": str(df['Date'].min().date()),
            "end": str(df['Date'].max().date()),
            "n_samples": len(df)
        }
    }
 
    result_path = os.path.join(OUTPUT_DIR, "prediction_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n    結果已儲存：{result_path}")
 
    # ── 特徵重要性CSV
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    feature_importance.to_csv(fi_path, index=False)
    print(f"    特徵重要性：{fi_path}")
 
    print(f"\n  ✅ 全部完成！請查看 {OUTPUT_DIR}/ 資料夾")
    return results
 
 
if __name__ == "__main__":
    results = main()