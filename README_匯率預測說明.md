# USD/TWD 匯率機器學習預測系統

---

## 一、研究目標

利用機器學習方法，結合**經濟基本面**與**技術分析**兩大框架，預測 **2025 年 5 月 29 日** 的台幣兌美元（USD/TWD）收盤匯率。

---
## 二、模型架構

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

## 三、如何執行

```bash
# 安裝依賴
pip install scikit-learn numpy pandas matplotlib scipy

# 執行預測
python3 usd_twd_predict_shap.py

# 輸出檔案
# output