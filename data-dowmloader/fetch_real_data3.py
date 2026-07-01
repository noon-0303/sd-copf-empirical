"""
從 Yahoo Finance 下載台灣（^TWII）、上海（000001.SS）、美國（^GSPC）日收盤價，
計算對數報酬率，取三市場共同交易日，存成 real_data3.xlsx。

輸出格式（每個 sheet 一個市場）：
  return1  shape (T, 1)：台灣 (TWII)
  return2  shape (T, 1)：上海 (SSE)
  return3  shape (T, 1)：美國 (GSPC)

日期範圍：2017-01-04 ~ 2024-09-25（與 real_data.xlsx 相同）
"""

import numpy as np
import pandas as pd
import yfinance as yf

START = '2016-12-30'
END   = '2024-09-25'
OUT   = r'C:\Users\user\PycharmProjects\sd-copf\real_data3.xlsx'

TICKERS = {
    'TWII': '^TWII',
    'SSE':  '000001.SS',
    'GSPC': '^GSPC',
}

rets = {}
for name, ticker in TICKERS.items():
    print(f'下載 {ticker} ({name}) ...')
    raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    ret = np.log(raw['Close'] / raw['Close'].shift(1)).dropna().squeeze()
    ret.name = name
    rets[name] = ret

# 取三市場共同交易日
df = pd.concat(rets.values(), axis=1, join='inner')
df = df.loc['2017-01-04':'2024-09-25'].dropna()

print(f'\n共同交易日筆數：{len(df)}')
print(f'日期區間：{df.index[0].date()} ~ {df.index[-1].date()}')
print(df.head(3))

with pd.ExcelWriter(OUT, engine='openpyxl') as writer:
    pd.DataFrame(df['TWII'].values).to_excel(writer, sheet_name='return1', index=False, header=False)
    pd.DataFrame(df['SSE'].values).to_excel(writer, sheet_name='return2', index=False, header=False)
    pd.DataFrame(df['GSPC'].values).to_excel(writer, sheet_name='return3', index=False, header=False)

print(f'\n已儲存至 {OUT}')
print('  return1 (T, 1) ← 台灣 (TWII)')
print('  return2 (T, 1) ← 上海 (SSE)')
print('  return3 (T, 1) ← 美國 (GSPC)')
