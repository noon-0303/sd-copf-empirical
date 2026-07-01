"""
以三國共同交易日（TW+SSE+GSPC 交集）為唯一基準，
重建所有實證資料檔，確保每個 xlsx 的 T 完全一致。

輸出（皆為 1 欄、無標題、shape (T,1)）：
  real_data3.xlsx      return1=TW, return2=SSE, return3=GSPC
  real_data3_tw.xlsx   return1=TW
  real_data3_sse.xlsx  return1=SSE
  real_data3_gspc.xlsx return1=GSPC
  real_data2-1.xlsx    return1=TW, return2=GSPC
  real_data2-2.xlsx    return1=TW, return2=SSE
"""

import numpy as np
import pandas as pd
import yfinance as yf

START = '2016-12-30'
END   = '2021-12-31'

OUT_DIR = r'C:\Users\user\PycharmProjects\sd-copf\Empirical'

# ── 1. 下載原始資料 ───────────────────────────────────────────────────────────
print('下載 ^TWII ...')
tw_raw  = yf.download('^TWII',      start=START, end=END, auto_adjust=True, progress=False)
print('下載 000001.SS ...')
sse_raw = yf.download('000001.SS',  start=START, end=END, auto_adjust=True, progress=False)
print('下載 ^GSPC ...')
us_raw  = yf.download('^GSPC',      start=START, end=END, auto_adjust=True, progress=False)

tw_ret  = np.log(tw_raw['Close']  / tw_raw['Close'].shift(1)).dropna().squeeze()
sse_ret = np.log(sse_raw['Close'] / sse_raw['Close'].shift(1)).dropna().squeeze()
us_ret  = np.log(us_raw['Close']  / us_raw['Close'].shift(1)).dropna().squeeze()

tw_ret.name  = 'TW'
sse_ret.name = 'SSE'
us_ret.name  = 'GSPC'

# ── 2. 三國共同交易日 ─────────────────────────────────────────────────────────
df = pd.concat([tw_ret, sse_ret, us_ret], axis=1, join='inner')
df = df.loc['2017-01-04':'2021-12-31'].dropna()

T = len(df)
print(f'\n三國共同交易日：{T} 筆')
print(f'日期區間：{df.index[0].date()} ~ {df.index[-1].date()}')

tw  = df['TW'].values
sse = df['SSE'].values
gs  = df['GSPC'].values

def save(path, sheets: dict):
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        for sh, arr in sheets.items():
            pd.DataFrame(arr.reshape(-1, 1)).to_excel(w, sheet_name=sh,
                                                       index=False, header=False)
    print(f'  儲存 {path}  ({T} 筆)')

# ── 3. 輸出所有檔案 ───────────────────────────────────────────────────────────
save(f'{OUT_DIR}\\real_data3.xlsx',
     {'return1': tw, 'return2': sse, 'return3': gs})

save(f'{OUT_DIR}\\real_data3_tw.xlsx',   {'return1': tw})
save(f'{OUT_DIR}\\real_data3_sse.xlsx',  {'return1': sse})
save(f'{OUT_DIR}\\real_data3_gspc.xlsx', {'return1': gs})

save(f'{OUT_DIR}\\real_data2-1.xlsx',
     {'return1': tw, 'return2': gs})

save(f'{OUT_DIR}\\real_data2-2.xlsx',
     {'return1': tw, 'return2': sse})

print('\n完成。所有檔案 T =', T)
