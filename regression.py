import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

def run_regression_analysis(csv_file):
    # 1. 載入資料
    df = pd.read_csv(csv_file)
    
    # 2. 定義特徵 (X) 與 目標 (y)
    # 我們選取幾個關鍵的基本面與技術面欄位
    features = ['fed_rate', 'tw_rate', 'rate_diff', 'dxy','foreign_net', 'VIXCLS', '^TWII', 'export']
    
    # 確保這些欄位都在 df 裡面
    X = df[features]
    y = df['USDTWD']
    
    # 3. 分割訓練集與測試集 (80% 訓練, 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. 建立線性回歸模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. 進行預測
    y_pred = model.predict(X_test)
    
    # 6. 輸出評估指標
    print("=== 線性回歸評估結果 ===")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R-squared: {r2_score(y_test, y_pred):.4f}")
    
    # 7. 顯示係數 (Coefficients) - 了解誰影響最大
    coeff_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
    print("\n=== 特徵係數 (影響權重) ===")
    print(coeff_df.sort_values(by='Coefficient', ascending=False))

    # 8. (選用) 使用 statsmodels 獲取更詳細的統計報告 (如 P-value)
    X_stat = sm.add_constant(X_train) # 加入截距項
    stat_model = sm.OLS(y_train, X_stat).fit()
    print("\n=== 詳細統計報告 (OLS Summary) ===")
    print(stat_model.summary())

if __name__ == "__main__":
    # 請確保檔案名稱與你輸出的 CSV 一致
    current_time = datetime.now().strftime('%Y%m%d')
    latest_file = f"USDTWD_Analysis_{current_time}.csv"
    run_regression_analysis(latest_file)