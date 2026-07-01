"""
下載台灣 (^TWII) 與美國 (^GSPC) 日收盤資料，
計算對數報酬率與 EWMA 變異數（λ=0.94），
存成 real_data.xlsx（4 sheets: return1/return2/volatility1/volatility2）
以供 MSAEtb - test.py 使用。
"""

import numpy as np
import pandas as pd
import yfinance as yf

START = '2017-01-01'
END   = '2024-12-31'
LAM   = 0.94          # RiskMetrics EWMA decay factor

OUTPUT = r'C:\Users\user\PycharmProjects\sd-copf\real_data.xlsx'


def ewma_var(r: np.ndarray, lam: float) -> np.ndarray:
    n = len(r)
    v = np.empty(n)
    v[0] = r[0] ** 2
    for t in range(1, n):
        v[t] = lam * v[t - 1] + (1 - lam) * r[t] ** 2
    return v


print("下載資料中...")
tw_raw = yf.download('^TWII', start=START, end=END, auto_adjust=True, progress=False)
us_raw = yf.download('^GSPC', start=START, end=END, auto_adjust=True, progress=False)

tw_close = tw_raw['Close'].squeeze()
us_close = us_raw['Close'].squeeze()

tw_ret = np.log(tw_close / tw_close.shift(1)).dropna()
us_ret = np.log(us_close / us_close.shift(1)).dropna()

# 對齊共同交易日
df = pd.DataFrame({'tw': tw_ret, 'us': us_ret}).dropna()
print(f"共同交易日總數：{len(df)}")

tw_r = df['tw'].values
us_r = df['us'].values
tw_v = ewma_var(tw_r, LAM)
us_v = ewma_var(us_r, LAM)

T_USE = len(tw_r)
print(f"\n使用全部 {T_USE} 個共同交易日")
print(f"  日期範圍：{df.index[0].date()} ~ {df.index[-1].date()}")
print(f"  TW 報酬率：mean={tw_r.mean():.6f},  std={tw_r.std():.6f}")
print(f"  US 報酬率：mean={us_r.mean():.6f},  std={us_r.std():.6f}")
print(f"  TW EWMA 變異：mean={tw_v.mean():.2e}, max={tw_v.max():.2e}")
print(f"  US EWMA 變異：mean={us_v.mean():.2e}, max={us_v.max():.2e}")

print(f"\n儲存至 {OUTPUT} ...")
with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
    pd.DataFrame(tw_r.reshape(-1, 1)).to_excel(writer, sheet_name='return1',    index=False, header=False)
    pd.DataFrame(us_r.reshape(-1, 1)).to_excel(writer, sheet_name='return2',    index=False, header=False)
    pd.DataFrame(tw_v.reshape(-1, 1)).to_excel(writer, sheet_name='volatility1', index=False, header=False)
    pd.DataFrame(us_v.reshape(-1, 1)).to_excel(writer, sheet_name='volatility2', index=False, header=False)

print("完成！")
