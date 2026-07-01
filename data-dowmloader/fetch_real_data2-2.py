"""
從 Yahoo Finance 下載台灣（^TWII）與上海（000001.SS）日收盤價，
計算對數報酬率，存成 real_data2.xlsx（格式同 real_data.xlsx）。

日期範圍：2017-01-04 ~ 2024-09-25（與 real_data.xlsx 相同）
輸出格式：
  return1 — TWII 對數報酬率，shape (T, 1)，無標題
  return2 — 上海綜合指數對數報酬率，shape (T, 1)，無標題
"""

import numpy as np
import pandas as pd
import yfinance as yf

START = '2016-12-30'   # 多抓幾天以確保報酬率計算不遺漏首日
END   = '2024-09-25'
OUT   = r'C:\Users\user\PycharmProjects\sd-copf\real_data2.xlsx'

print('下載 ^TWII ...')
twii_raw = yf.download('^TWII', start=START, end=END, auto_adjust=True, progress=False)
twii_ret = np.log(twii_raw['Close'] / twii_raw['Close'].shift(1)).dropna().squeeze()
twii_ret.name = 'TWII'

print('下載 000001.SS (上海綜合指數) ...')
sse_raw  = yf.download('000001.SS', start=START, end=END, auto_adjust=True, progress=False)
sse_ret  = np.log(sse_raw['Close'] / sse_raw['Close'].shift(1)).dropna().squeeze()
sse_ret.name = 'SSE'

# 取共同交易日後，再截取 2017-01-04 ~ 2024-09-25
df = pd.concat([twii_ret, sse_ret], axis=1, join='inner')
df = df.loc['2017-01-04':'2024-09-25']
df = df.dropna()

print(f'共同交易日筆數：{len(df)}')
print(f'日期區間：{df.index[0].date()} ~ {df.index[-1].date()}')
print(df.head(3))

# 存成 Excel（無標題、各一欄）
with pd.ExcelWriter(OUT, engine='openpyxl') as writer:
    pd.DataFrame(df['TWII'].values).to_excel(writer, sheet_name='return1', index=False, header=False)
    pd.DataFrame(df['SSE'].values).to_excel(writer, sheet_name='return2', index=False, header=False)

print(f'\n已儲存至 {OUT}')
