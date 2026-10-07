# USD/TWD 匯率機器學習預測系統
## 作業報告 | 2025/05/20 截止

---

## 一、研究目標

利用機器學習方法，結合**經濟基本面**與**技術分析**兩大框架，預測 **2025 年 5 月 29 日** 的台幣兌美元（USD/TWD）收盤匯率。

---

## 二、理論架構與特徵設計

### 2.1 經濟基本面特徵（Economic Fundamentals）

根據國際經濟學理論，影響台幣匯率的主要經濟變數如下：

| 特徵 | 理論依據 | 預期方向 |
|------|----------|----------|
| 美台利差（Fed Rate - CBC Rate） | 利率平價理論（UIP）：高利差貨幣傾向升值 | 利差↑ → 台幣貶值（USD/TWD↑） |
| 美元指數代理（DXY Proxy） | DXY 為全球美元強弱指標 | DXY↑ → 台幣貶值 |
| VIX 恐慌指數 | 風險趨避時資金流向美元避險 | VIX↑ → 台幣貶值 |
| 台灣出口景氣 | 台灣出口占GDP約70%，外匯供給關鍵 | 出口↑ → 台幣升值（USD/TWD↓） |
| 外資股市流動 | 外資買超台股時需換匯 | 外資流入↑ → 台幣升值 |
| 購買力平價偏差（PPP） | 台幣長期被低估（大麥克指數） | 提供均值回歸信號 |
| 季節性（農曆年、半導體庫存週期） | 台灣特有結構性因素 | 農曆年前 → 換匯需求↑ |

**特別說明：** 2025 年 4-5 月，因川普關稅政策引發外資資金大規模自美股撤出回流亞洲，加上台灣出口商大量結售美元，導致台幣在短期內大幅升值超過 4%，為本模型的重要情境因素。

### 2.2 技術分析特徵（Technical Analysis）

| 指標 | 計算方式 | 用途 |
|------|----------|------|
| 移動平均（MA5/10/20/60） | 簡單移動平均 | 趨勢方向與強度 |
| MACD | EMA12 - EMA26，Signal=EMA9 | 動能轉折點 |
| RSI（14日） | 相對強弱指標 | 超買超賣判斷 |
| Bollinger Bands | MA20 ± 2σ | 波動率與價格位置 |
| 報酬率（1/3/5/10/20日） | 日對數報酬率 | 動能效應捕捉 |
| 波動度（10/20日） | 歷史波動率 | 市場不確定性量化 |
| 偏離率（乖離率） | (P - MA20) / MA20 | 均值回歸信號 |

---

## 三、模型架構

### 集成學習（Ensemble Learning）

本研究採用**加權集成**策略，依各模型驗證集 MAE 倒數計算權重：

```
集成預測 = Σ (weight_i × model_i_prediction)
weight_i = (1/MAE_i) / Σ(1/MAE_j)
```

| 模型 | 特性 | 適用場景 |
|------|------|----------|
| **Random Forest** | 非線性、防過擬合 | 捕捉複雜交互效應 |
| **Gradient Boosting** | 序列殘差學習 | 精細的趨勢修正 |
| **Ridge Regression** | L2 正則化線性模型 | 利差等線性經濟關係 |
| **ElasticNet** | L1+L2 正則化 | 特徵稀疏選擇 |

### 時間序列交叉驗證（Time-Series CV）

使用 `TimeSeriesSplit` 確保未來資訊不洩漏至訓練集，共 5 折驗證。

---

## 四、預測結果

### 2025/05/29 USD/TWD 預測

| 模型 | 預測值 |
|------|--------|
| Random Forest | 31.08 |
| Gradient Boosting | 31.13 |
| Ridge Regression | 31.44 |
| ElasticNet | 31.32 |
| **集成模型（最終預測）** | **31.26** |
| 信賴區間（±1 RMSE） | [30.93, 31.59] |

### 模型評估指標

| 指標 | 測試集 | CV 平均 |
|------|--------|---------|
| MAE | 0.2641 | 0.2971 |
| RMSE | 0.3274 | 0.3722 |
| R² | 0.8617 | — |
| MAPE | 0.86% | 0.97% |

**解讀：** MAPE ≈ 0.86% 表示平均預測誤差不超過匯率的 1%，對於匯率短期預測而言屬於良好表現。

---

## 五、關鍵特徵解讀

依 Random Forest 特徵重要性排名，最重要的預測因子為：

1. **技術面**：移動平均、EMA、近期報酬率
2. **經濟面**：DXY 代理、聯準會利率、台美利差
3. **情緒面**：VIX 恐慌指數
4. **結構面**：外資流動、季節效應

---

## 六、模型限制與風險

1. **黑天鵝事件：** 模型無法預測突發的政治、地緣衝突、央行干預等事件
2. **非定常性：** 匯率時間序列受到結構性斷點影響（如 QE、升息週期切換）
3. **資料代理誤差：** 部分經濟特徵以代理變數估算，存在測量誤差
4. **有效市場假說：** 強式有效市場理論認為技術分析無法持續獲利，但短期動能效應仍具預測價值

---

## 七、如何執行

```bash
# 安裝依賴
pip install scikit-learn numpy pandas matplotlib scipy

# 執行預測
python3 usd_twd_predict.py

# 輸出檔案
# output/usd_twd_prediction.png     → 4張圖表
# output/prediction_results.json    → 完整預測結果
# output/feature_importance.csv     → 特徵重要性
```

### 進階：使用真實資料

在 `fetch_usd_twd_from_stooq()` 函式中，可替換為以下真實資料來源：
- **台灣央行：** https://www.cbc.gov.tw（歷史匯率月資料）
- **FRED API：** https://fred.stlouisfed.org（美國利率、DXY）
- **Yahoo Finance（yfinance）：** `pip install yfinance`，代碼 `USDTWD=X`
- **台灣股市資訊：** https://www.twse.com.tw（外資買超資料）

---

## 參考文獻

1. Frankel & Rose (1995). Empirical research on nominal exchange rates. *Handbook of International Economics*
2. Meese & Rogoff (1983). Empirical exchange rate models of the seventies. *Journal of International Economics*
3. Ni et al. (2024). Enhancing Exchange Rate Forecasting with Explainable Deep Learning Models. *arXiv:2410.19241*
4. 台灣央行 (2025). 有關國內匯市及美債等議題之說明. https://www.cbc.gov.tw
5. 今周刊 (2025). 台幣對美金匯率被低估55%？「台灣病」分析
