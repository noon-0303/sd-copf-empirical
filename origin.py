import numpy as np
import cupy as cp
import pandas as pd
import unicodedata
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# 0.1 載入資料 ─────────────────────────────────────────────────────────────────

DATA_PATH = r'C:\Users\user\PycharmProjects\sd-copf\Empirical\data\real_data3_tw.xlsx'
data  = pd.read_excel(DATA_PATH, sheet_name='return1', header=None).values
y_vec = data[:, 0]
T     = len(y_vec)

LABEL   = '資產 '
var_emp = np.var(y_vec, ddof=1)


# 0.2 超參數 ──────────────────────────────────────────────────────────────────

N      = 300
k      = 5
h      = T * 1e-4
a_coef = np.sqrt(1 - h ** 2)
d      = 1 / 10

KAPPA_SCALE = 1.0
SIGMA_SCALE = 1.0
MIN_VAR     = 1e-4

FCST_HORIZONS = [1, 5, 10, 22]
WIN_OOS_MAX   = 120
RV_WIN        = 1


def _est_sv_params(y_s, v_s):
    mu_e    = float(np.mean(y_s))
    r       = y_s - mu_e
    proxy   = r ** 2
    lam     = 0.94
    rv      = pd.Series(proxy).ewm(alpha=1 - lam).mean().values
    rv      = np.clip(rv, MIN_VAR, None)
    theta_e = float(max(np.mean(proxy), MIN_VAR))
    x_ar    = rv[:-1]
    yv      = rv[1:]
    b       = np.cov(x_ar, yv)[0, 1] / np.var(x_ar)
    kappa_e = float(np.clip(1.0 - b, 1e-3, 0.999))
    ar1_int = float(np.mean(yv) - b * np.mean(x_ar))
    u       = yv - ar1_int - b * x_ar
    sigma_e = float(np.sqrt(np.mean(u ** 2 / np.clip(x_ar, MIN_VAR, None))))
    sigma_e = max(sigma_e, 1e-4)
    eps1     = r[1:] / np.sqrt(np.clip(x_ar, MIN_VAR, None))
    eta      = u / (sigma_e * np.sqrt(np.clip(x_ar, MIN_VAR, None)))
    corr_lev = np.corrcoef(eps1, eta)[0, 1]
    rho_e    = float(np.clip(corr_lev if np.isfinite(corr_lev) else 0.0, -0.999, 0.999))
    kappa_e *= KAPPA_SCALE
    sigma_e  = max(sigma_e * SIGMA_SCALE, 1e-4)
    if 2 * kappa_e * theta_e < sigma_e ** 2:
        sigma_e = float(np.sqrt(2 * kappa_e * theta_e) * 0.95)
    centers = np.array([mu_e, kappa_e, theta_e, sigma_e, rho_e])
    v_arr   = np.array([
        max(abs(mu_e), np.sqrt(theta_e) * 0.1) * d,
        kappa_e * d,
        theta_e * d,
        max(sigma_e, 1e-4) * d,
        max(abs(rho_e), 0.1) * d
    ])
    return centers, v_arr


init_mean, v_arr = _est_sv_params(y_vec, max(var_emp, MIN_VAR))


def normpdf_gpu(x, mu, sigma):
    return cp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * cp.sqrt(2 * cp.pi))


theta_s = np.zeros((T, k))
x_s     = np.zeros(T)


# 0.3 迴圈內變數 ──────────────────────────────────────────────────────────────

x_arr    = cp.zeros((T, N))
mu_arr   = cp.zeros((T, N))
theta    = cp.zeros((T, N, k))
m_smooth = cp.zeros((T, N, k))
w        = cp.zeros((T, N))

v1    = v_arr.copy()
v_all = cp.array(v1 ** 2 * d)

# 初始 x（normal，負值反射）
x_arr[0] = cp.random.normal(var_emp, var_emp * d, N)
neg_x0   = x_arr[0] < 0
x_arr[0] = cp.where(neg_x0, 2 * var_emp - x_arr[0], x_arr[0])

mu_arr[0] = x_arr[0]
x_s[0]    = float(cp.mean(x_arr[0]))

# theta 1：mu（常態）
theta[0, :, 0] = init_mean[0] + v1[0] * cp.random.randn(N)

# theta 2：kappa（gamma）
kappa_init     = init_mean[1]
alpha_k        = kappa_init ** 2 / v_arr[1] ** 2
beta_k         = v_arr[1] ** 2 / kappa_init
theta[0, :, 1] = cp.maximum(cp.random.gamma(alpha_k, beta_k, N), 1e-6)

# theta 3：theta/長期變異數（gamma）
alpha2         = init_mean[2] ** 2 / v1[2] ** 2
beta2          = v1[2] ** 2 / init_mean[2]
theta[0, :, 2] = cp.maximum(cp.random.gamma(alpha2, beta2, N), 1e-6)

# theta 4：sigma（gamma）
alpha3         = init_mean[3] ** 2 / v1[3] ** 2
beta3          = v1[3] ** 2 / init_mean[3]
theta[0, :, 3] = cp.maximum(cp.random.gamma(alpha3, beta3, N), 1e-6)

# theta 5：rho（uniform）
_a_rho         = float(init_mean[4] - np.sqrt(3) * v_arr[4])
_b_rho         = float(init_mean[4] + np.sqrt(3) * v_arr[4])
theta[0, :, 4] = cp.clip(cp.random.uniform(_a_rho, _b_rho, N), -0.999, 0.999)

theta_s[0] = cp.asnumpy(cp.mean(theta[0], axis=0))

# 0.4 初始權重
w[0] = 1.0 / N

# 0.5 預先產生隨機數
r2 = cp.random.randn(T, N)

logL = 0.0


# ── 時間迴圈 ─────────────────────────────────────────────────────────────────
for t in range(1, T):

    # 1.1 加權平均與平滑參數
    theta_mean    = cp.sum(w[t-1, :, None] * theta[t-1], axis=0)
    m_smooth[t-1] = a_coef * theta[t-1] + (1 - a_coef) * theta_mean

    # 1.2 估計波動度
    mu_arr[t] = (x_arr[t-1]
                 + m_smooth[t-1, :, 1] * (m_smooth[t-1, :, 2] - x_arr[t-1]))

    # 1.3 Likelihood
    y_t       = float(y_vec[t])
    std_denom = cp.sqrt(cp.maximum(mu_arr[t], 1e-10))
    p         = normpdf_gpu(y_t, m_smooth[t-1, :, 0] - mu_arr[t] / 2, std_denom)
    p         = cp.nan_to_num(p)
    g         = w[t-1] * p
    g_sum     = cp.sum(g)
    logL     += float(cp.log(g_sum)) if float(g_sum) > 0 else 0.0
    g         = g / g_sum if float(g_sum) > 0 else cp.full(N, 1.0 / N)

    # 2.1 重抽樣
    rs          = np.random.choice(N, N, replace=True, p=cp.asnumpy(g))
    x_arr[t-1]  = x_arr[t-1, rs]
    theta[t-1]  = theta[t-1, rs, :]

    # 2.2 重新計算平滑參數與預期波動度
    theta_mean    = cp.mean(theta[t-1], axis=0)
    m_smooth[t-1] = a_coef * theta[t-1] + (1 - a_coef) * theta_mean
    v             = cp.maximum(cp.var(theta[t-1], axis=0, ddof=1), v_all)
    mu_arr[t]     = (x_arr[t-1]
                     + m_smooth[t-1, :, 1] * (m_smooth[t-1, :, 2] - x_arr[t-1]))
    neg_mu_idx    = mu_arr[t] < 0
    mu_arr[t]     = cp.abs(mu_arr[t])

    # 2.3 參數更新
    # theta 1：mu（常態）
    theta[t, :, 0] = (m_smooth[t-1, :, 0]
                      + cp.sqrt(h ** 2 * v[0]) * cp.random.randn(N))

    # theta 2：kappa（gamma）
    ms1            = cp.maximum(m_smooth[t-1, :, 1], 1e-10)
    var1t          = cp.maximum(h ** 2 * v[1], 1e-20)
    alpha1t        = ms1 ** 2 / var1t
    beta1t         = var1t / ms1
    theta[t, :, 1] = cp.maximum(cp.random.gamma(alpha1t, beta1t, N), 1e-6)

    # theta 3：theta/長期變異數（gamma）
    ms2            = cp.maximum(m_smooth[t-1, :, 2], 1e-10)
    var2           = cp.maximum(h ** 2 * v[2], 1e-20)
    alpha2t        = ms2 ** 2 / var2
    beta2t         = var2 / ms2
    theta[t, :, 2] = cp.random.gamma(alpha2t, beta2t, N)

    # theta 4：sigma（gamma）
    ms3            = cp.maximum(m_smooth[t-1, :, 3], 1e-10)
    var3           = cp.maximum(h ** 2 * v[3], 1e-20)
    alpha3t        = ms3 ** 2 / var3
    beta3t         = var3 / ms3
    theta[t, :, 3] = cp.random.gamma(alpha3t, beta3t, N)

    # theta 5：rho（uniform）
    sd4t           = cp.sqrt(h ** 2 * v[4])
    _a_rho_t       = m_smooth[t-1, :, 4] - cp.sqrt(3.0) * sd4t
    _b_rho_t       = m_smooth[t-1, :, 4] + cp.sqrt(3.0) * sd4t
    u_rho          = cp.random.uniform(0, 1, N)
    theta[t, :, 4] = cp.clip(_a_rho_t + (_b_rho_t - _a_rho_t) * u_rho, -0.999, 0.999)

    # 2.4 狀態更新
    x_prev   = cp.maximum(x_arr[t-1], 1e-10)
    r1       = (y_t - (theta[t, :, 0] - 0.5 * x_arr[t-1])) / cp.sqrt(x_prev)
    x_arr[t] = (x_arr[t-1]
                + theta[t, :, 1] * (theta[t, :, 2] - x_arr[t-1])
                + theta[t, :, 3] * cp.sqrt(x_prev)
                * (theta[t, :, 4] * r1
                   + cp.sqrt(1 - theta[t, :, 4] ** 2) * r2[t]))

    neg_idx   = x_arr[t] < 0
    x_arr[t]  = cp.abs(x_arr[t])

    # 3.1 權重更新
    p1 = normpdf_gpu(y_t,
                     theta[t, :, 0] - x_arr[t] / 2,
                     cp.sqrt(cp.maximum(x_arr[t], 1e-10)))
    p2 = normpdf_gpu(y_t,
                     m_smooth[t-1, :, 0] - mu_arr[t] / 2,
                     cp.sqrt(cp.maximum(mu_arr[t], 1e-10)))

    w[t]             = cp.nan_to_num(p1 / p2)
    w[t, neg_idx]    = 0.0
    w[t, neg_mu_idx] = 0.0
    w_sum            = cp.sum(w[t])
    w[t]             = w[t] / w_sum if float(w_sum) > 0 else cp.full(N, 1.0 / N)

    theta_s[t] = cp.asnumpy(cp.mean(theta[t], axis=0))
    x_s[t]     = float(cp.mean(x_arr[t]))

# 儲存最終粒子雲（供殘差診斷使用）
theta_last = cp.asnumpy(theta[T-1])   # (N, k)
x_last     = cp.asnumpy(x_arr[T-1])   # (N,)


# ── Log-Likelihood ────────────────────────────────────────────────────────────
print(f'\n── {LABEL} Log-Likelihood ──')
print(f'  logL: {logL:.4f}')

# ── 估計參數（時間平均）─────────────────────────────────────────────────────
param_labels   = ['mu', 'kappa', 'theta', 'sigma', 'rho']
theta_estimate = theta_s[T-1]
print(f'\n── {LABEL} 估計參數 ──')
print(f'    {"":8s}  {LABEL:>14s}')
for pi, pn in enumerate(param_labels):
    print(f'    {pn:8s}: {theta_estimate[pi]:14.6f}')


# ── 繪圖 ─────────────────────────────────────────────────────────────────────
t_arr = np.arange(T)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(t_arr, y_vec)
axes[0].set_title('Returns')
axes[1].plot(t_arr, x_s, label='Filtered')
axes[1].set_title('Volatility')
axes[1].legend()
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ── 殘差診斷與風險指標 ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

mu_hat  = theta_s[:, 0] - x_s / 2
sig_hat = np.sqrt(np.maximum(x_s, 1e-10))
z_res   = (y_vec - mu_hat) / sig_hat
u_pit   = stats.norm.cdf(z_res)


# ── 殘差診斷 ─────────────────────────────────────────────────────────────────
ks_res = stats.kstest(u_pit, 'uniform')
lb_p   = acorr_ljungbox(z_res,    lags=[20], return_df=True)['lb_pvalue'].iloc[-1]
lb2_p  = acorr_ljungbox(z_res**2, lags=[20], return_df=True)['lb_pvalue'].iloc[-1]

print(f'\n── 殘差診斷（{LABEL}）──')
print(f'  {LABEL}:')
print(f'    KS  (PIT vs U[0,1]):  stat={ks_res.statistic:.4f}  p={ks_res.pvalue:.4f}'
      + ('  （拒絕）' if ks_res.pvalue < 0.05 else '  （未拒絕）'))
print(f'    LB(20)  殘差自相關:    p={lb_p:.4f}'
      + ('  （拒絕）' if lb_p < 0.05 else '  （未拒絕）'))
print(f'    LB²(20) ARCH 效果:     p={lb2_p:.4f}'
      + ('  （拒絕）' if lb2_p < 0.05 else '  （未拒絕）'))


# ── 條件 VaR / ES（近似分布）─────────────────────────────────────────────────
VaR_LEVELS = [0.01, 0.05]

print(f'\n── {LABEL} 條件 VaR / ES（近似分布）──')
print(f'    {"":10s}  {LABEL+" 均值":>14s}')
for alpha in VaR_LEVELS:
    pct      = int((1 - alpha) * 100)
    z_alpha  = np.quantile(z_res, alpha)
    mask     = z_res <= z_alpha
    es_z     = np.mean(z_res[mask]) if mask.any() else z_alpha
    var_mean = (-(mu_hat + z_alpha * sig_hat)).mean()
    es_mean  = (-(mu_hat + es_z   * sig_hat)).mean()
    print(f'    VaR {pct}%:  {var_mean:12.6f}')
    print(f'    ES  {pct}%:  {es_mean:12.6f}')


# ── 輔助函數 ──────────────────────────────────────────────────────────────────

def _dw(s):
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in str(s))

def _lj(s, width): return str(s) + ' ' * max(0, width - _dw(str(s)))
def _rj(s, width): return ' ' * max(0, width - _dw(str(s))) + str(s)
def _fl(v):        return f'{v:.4f}' if not np.isnan(v) else 'nan'
def _sig(p):       return '*' if (not np.isnan(p) and p < 0.05) else ' '


def _kupiec_pof(hit, alpha):
    T_, x = len(hit), int(hit.sum())
    p = x / T_
    if x == 0 or x == T_:
        return np.nan, np.nan
    lr = -2.0 * (x * np.log(alpha / p) + (T_ - x) * np.log((1 - alpha) / (1 - p)))
    return lr, float(1.0 - stats.chi2.cdf(lr, df=1))


def _christoffersen_ind(hit):
    v   = hit.astype(int)
    n00 = int(((v[:-1] == 0) & (v[1:] == 0)).sum())
    n01 = int(((v[:-1] == 0) & (v[1:] == 1)).sum())
    n10 = int(((v[:-1] == 1) & (v[1:] == 0)).sum())
    n11 = int(((v[:-1] == 1) & (v[1:] == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi   = (n01 + n11) / len(v)
    if pi in (0.0, 1.0) or pi01 in (0.0, 1.0) or pi11 in (0.0, 1.0):
        return np.nan, np.nan
    lr = -2.0 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
        - n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )
    return lr, float(1.0 - stats.chi2.cdf(lr, df=1))


def _dm_test(loss1, loss2, h_horizon=1):
    """
    Diebold-Mariano (1995) + Harvey-Leybourne-Newbold (1997) 小樣本修正。
    H₀: E[loss1 − loss2] = 0
    Returns: DM stat, p-value (two-sided), p-value (one-sided: loss1 < loss2)
    """
    d    = loss1 - loss2
    d    = d[~np.isnan(d)]
    n    = len(d)
    if n < 2:
        return np.nan, np.nan, np.nan
    d_bar  = np.mean(d)
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv    = gamma0
    for lag in range(1, h_horizon):
        w_lag    = 1.0 - lag / h_horizon
        gamma_l  = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        lrv     += 2.0 * w_lag * gamma_l
    lrv    = max(lrv, 1e-20)
    dm     = d_bar / np.sqrt(lrv / n)
    p_two  = 2.0 * float(stats.t.sf(abs(dm), df=n - 1))
    p_less = float(stats.t.cdf(dm, df=n - 1))   # H₁: loss1 < loss2
    return float(dm), p_two, p_less




# ── SV 參數有效性檢定 ────────────────────────────────────────────────────────
#
# def _ttest_gt0(arr):
#     t, p = stats.ttest_1samp(arr, popmean=0, alternative='greater')
#     return float(np.mean(arr)), float(t), float(p)
#
# PIDX = [1,       2,       3      ]
# PLAB = ['κ > 0', 'θ > 0', 'σ > 0']
# PDSC = ['均值回歸為正', '長期變異數為正', '波動率為正']
# PW   = [16,      10,      8,     7]
#
# print(f'\n── SV 參數有效性檢定（{LABEL}）──')
# print('    H₀: 各參數 ≤ 0；單側 t 檢定，* p < 0.05 拒絕 H₀（確認條件成立）')
# print(f'\n    {"":16s}  {f"─── {LABEL} ───":^27s}')
# print('    ' + '  '.join([
#     _rj('檢定', PW[0]), _rj('估計值', PW[1]), _rj('t-stat', PW[2]), _rj('p 值', PW[3]),
# ]))
# print('    ' + '  '.join('-' * ww for ww in PW))
#
# for pidx, plab, pdsc in zip(PIDX, PLAB, PDSC):
#     mv, tv, pv = _ttest_gt0(theta_last[:, pidx])
#     label = f'{plab}（{pdsc}）'
#     row = [
#         _lj(label,       PW[0]),
#         _rj(f'{mv:.6f}', PW[1]),
#         _rj(f'{tv:.4f}', PW[2]),
#         _rj(f'{pv:.4f}' + _sig(pv), PW[3]),
#     ]
#     print('    ' + '  '.join(row))
#
# feller = 2 * theta_last[:, 1] * theta_last[:, 2] - theta_last[:, 3] ** 2
# mf, tf, pf = _ttest_gt0(feller)
# feller_row = [
#     _lj('2κθ > σ²（Feller 條件）', PW[0]),
#     _rj(f'{mf:.6f}', PW[1]),
#     _rj(f'{tf:.4f}', PW[2]),
#     _rj(f'{pf:.4f}' + _sig(pf), PW[3]),
# ]
# print('    ' + '  '.join(feller_row))



# ─────────────────────────────────────────────────────────────────────────────
# ── Volatility Forecasting & QLIKE ───────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

PRED_PATH = r'C:\Users\user\PycharmProjects\sd-copf\Empirical\data\real_data3_tw_pred.xlsx'
y_pred = pd.read_excel(PRED_PATH, sheet_name='return1', header=None).values.ravel()
T_pred = len(y_pred)

kappa_hat = float(np.mean(theta_s[:, 1]))
long_hat  = float(np.mean(theta_s[:, 2]))
kappa_c   = float(np.clip(kappa_hat, 0.0, 1.0))

# ── A. 樣本內 1-step-ahead forecast（用於 IS QLIKE 對照）────────────────────
h_is    = np.empty(T)
h_is[0] = x_s[0]
for t in range(1, T):
    h_is[t] = x_s[t-1] + kappa_c * (long_hat - x_s[t-1])
h_is = np.maximum(h_is, 1e-10)

# VaR 分位數基準：以一步預測方差（與 OOS h_roll 同尺度）重新標準化 IS 殘差
# h_is[t] = x_s[t-1] + κ(θ−x_s[t-1])，與 h_roll 同為「看到 y 之前」的預測方差
mu_hat_pred  = theta_s[1:, 0] - h_is[1:] / 2
sig_hat_pred = np.sqrt(h_is[1:])
z_res_pred   = (y_vec[1:] - mu_hat_pred) / sig_hat_pred

mu_is    = float(np.mean(theta_s[:, 0]))
drift_is = mu_is - 0.5 * x_s[:-1]
z_is     = y_vec[1:] - drift_is
rv_is_m  = pd.Series(z_is ** 2).rolling(RV_WIN).mean().values

qlike_is   = float(np.nanmean(np.log(h_is[1:]) + rv_is_m / h_is[1:]))
h_const_is = float(np.nanmean(rv_is_m))

# ── B. 滾動 OOS 序列粒子濾波（固定參數，逐期以觀測值更新波動度狀態）────────
r2_oos       = np.random.randn(T_pred, N)
x_oos        = x_last.copy()            # (N,) IS 末期後驗狀態
x_oos_states = np.empty((T_pred, N))   # 各 OOS 期濾波狀態（觀測前）
h_roll       = np.empty(T_pred)

for t in range(T_pred):
    x_oos_states[t] = x_oos

    mu_next   = x_oos + theta_last[:, 1] * (theta_last[:, 2] - x_oos)
    h_roll[t] = float(np.mean(np.maximum(mu_next, 1e-10)))

    if t < T_pred - 1:
        sx   = np.sqrt(np.maximum(x_oos, 1e-8))
        r1   = (y_pred[t] - (theta_last[:, 0] - 0.5 * x_oos)) / sx
        rho  = theta_last[:, 4]
        x_oos = np.maximum(
            x_oos + theta_last[:, 1] * (theta_last[:, 2] - x_oos)
            + theta_last[:, 3] * sx
              * (rho * r1 + np.sqrt(np.maximum(1 - rho ** 2, 0)) * r2_oos[t]),
            1e-10)

mu_oos    = float(np.mean(theta_last[:, 0]))
drift_oos = mu_oos - 0.5 * np.mean(x_oos_states, axis=1)
z_pred    = y_pred - drift_oos
rv_pred_m = pd.Series(z_pred ** 2).rolling(RV_WIN).mean().values

# ── C. 樣本內 vs 樣本外 QLIKE 比較表 ────────────────────────────────────────
qlike_oos       = float(np.nanmean(np.log(h_roll) + rv_pred_m / h_roll))
qlike_bench_is  = np.log(h_const_is) + float(np.nanmean(rv_is_m  / h_const_is))
qlike_bench_oos = np.log(h_const_is) + float(np.nanmean(rv_pred_m / h_const_is))

print(f'\n── QLIKE 比較（realized proxy = {RV_WIN}-day rolling z²（去漂移殘差））──')
print(f'  {"":22s}  {LABEL:>12s}')
print('  ' + '-' * 40)
print(f'  {"IS  SV":22s}  {qlike_is:12.6f}')
print(f'  {"IS  常數基準":22s}  {qlike_bench_is:12.6f}')
print(f'  {"IS  差值(基準-模型)":22s}  {qlike_bench_is - qlike_is:12.6f}')
print('  ' + '-' * 40)
print(f'  {"OOS SV":22s}  {qlike_oos:12.6f}')
print(f'  {"OOS 常數基準(IS σ²)":22s}  {qlike_bench_oos:12.6f}')
print(f'  {"OOS 差值(基準-模型)":22s}  {qlike_bench_oos - qlike_oos:12.6f}')
print('  （差值 > 0 表示模型優於常數基準）')

# ── D. Diebold-Mariano 檢定（1-step IS & OOS）───────────────────────────────
loss_sv_is    = np.log(h_is[1:])    + rv_is_m   / h_is[1:]
loss_bench_is = np.log(h_const_is)  + rv_is_m   / h_const_is
loss_sv_oos   = np.log(h_roll)      + rv_pred_m  / h_roll
loss_bench_oos= np.log(h_const_is)  + rv_pred_m  / h_const_is

dm_is,  p2_is,  pl_is  = _dm_test(loss_sv_is,  loss_bench_is,  h_horizon=1)
dm_oos, p2_oos, pl_oos = _dm_test(loss_sv_oos, loss_bench_oos, h_horizon=1)

print('\n── Diebold-Mariano 檢定（SV vs 常數基準，QLIKE 損失差）──')
print('  H₀: 等預測精準度；H₁(less): SV 損失 < 基準損失（* p<0.05）')
print(f'  {"":12s}  {"DM-stat":>10s}  {"p(two)":>8s}  {"p(less)":>9s}')
print('  ' + '-' * 46)
print(f'  {"IS  (h=1)":12s}  {dm_is:10.4f}  {p2_is:8.4f}  {pl_is:8.4f}{_sig(pl_is)}')
print(f'  {"OOS (h=1)":12s}  {dm_oos:10.4f}  {p2_oos:8.4f}  {pl_oos:8.4f}{_sig(pl_oos)}')

# ─────────────────────────────────────────────────────────────────────────────
# ── Rolling h-step Forecast QLIKE（樣本外）────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

print('\n── Rolling h-step Forecast QLIKE & DM 檢定（樣本外，CIR 解析解）──')
print(f'  {"horizon (days)":>16s}  {"QLIKE":>12s}  {"有效期數":>8s}  {"DM-stat":>9s}  {"p(two)":>8s}  {"p(less)":>9s}')
print('  ' + '-' * 72)

for h_fc in FCST_HORIZONS:
    n_periods = T_pred - h_fc + 1
    if n_periods <= 0:
        continue

    st    = x_oos_states[:n_periods]                       # (n_periods, N)
    k_fc  = np.clip(theta_last[None, :, 1], 0.0, 1.0)     # (1, N)
    th    = theta_last[None, :, 2]                                    # (1, N)

    fcast  = th + (st - th) * (1 - k_fc) ** h_fc          # (n_periods, N)
    fh     = np.mean(np.maximum(fcast, 1e-10), axis=1)    # (n_periods,)

    rv_tgt     = rv_pred_m[h_fc-1:h_fc-1+n_periods]
    loss_sv_h  = np.log(fh)          + rv_tgt / fh
    loss_bch_h = np.log(h_const_is)  + rv_tgt / h_const_is
    q          = float(np.nanmean(loss_sv_h))
    dm_h, p2_h, pl_h = _dm_test(loss_sv_h, loss_bch_h, h_horizon=h_fc)
    print(f'  {h_fc:>16d}  {q:12.6f}  {n_periods:>8d}  {dm_h:9.4f}  {p2_h:8.4f}  {pl_h:8.4f}{_sig(pl_h)}')



# ─────────────────────────────────────────────────────────────────────────────
# ── VaR 回測（OOS）：Kupiec POF & Christoffersen CC ──────────────────────────
# z_alpha 由 IS 殘差估計（無前視），條件 VaR 來自 OOS 粒子濾波 1-step 預測
# ─────────────────────────────────────────────────────────────────────────────

WCOL = [14, 9, 7, 8, 7, 8, 7, 8, 7]
HCOL = ['資產', '違反/T', '違反率', 'LR_POF', 'p_POF', 'LR_ind', 'p_ind', 'LR_cc', 'p_cc']
SEP  = '  '

mu_hat_oos  = drift_oos
sig_hat_oos = np.sqrt(np.maximum(h_roll, 1e-10))

# print("IS  z_res q01/q05:", np.quantile(z_res, 0.01), np.quantile(z_res, 0.05))
# print("理論常態      :", stats.norm.ppf(0.01), stats.norm.ppf(0.05))
# print("IS  sig 均值:", np.mean(sig_hat), " OOS sig 均值:", np.mean(sig_hat_oos))
# print("h_roll 前/後段均值:", h_roll[:50].mean(), h_roll[-50:].mean())
print("OOS 報酬 std:", y_pred.std(), " IS 報酬 std:", y_vec.std())

for alpha in VaR_LEVELS:
    pct          = int((1 - alpha) * 100)
    z_alpha_is   = np.quantile(z_res_pred, alpha)   # 改用預測方差標準化分位數
    cond_VaR_oos = -(mu_hat_oos + z_alpha_is * sig_hat_oos)

    hit_oos = y_pred < -cond_VaR_oos
    x_cnt   = int(hit_oos.sum())
    p_hat   = x_cnt / T_pred
    lr_uc, p_uc   = _kupiec_pof(hit_oos, alpha)
    lr_ind, p_ind = _christoffersen_ind(hit_oos)
    lr_cc = (lr_uc + lr_ind) if not (np.isnan(lr_uc) or np.isnan(lr_ind)) else np.nan
    p_cc  = float(1.0 - stats.chi2.cdf(lr_cc, df=2)) if not np.isnan(lr_cc) else np.nan

    print(f'\n── OOS VaR 回測 {pct}%（α={alpha:.2f}）—— Kupiec POF & Christoffersen CC ──')
    print('    (* p<0.05 拒絕 H₀)')
    print('    ' + SEP.join(_rj(hh, ww) for hh, ww in zip(HCOL, WCOL)))
    print('    ' + SEP.join('-' * ww for ww in WCOL))

    row = [
        _lj(LABEL,               WCOL[0]),
        _rj(f'{x_cnt}/{T_pred}', WCOL[1]),
        _rj(f'{p_hat:.4f}',      WCOL[2]),
        _rj(_fl(lr_uc),          WCOL[3]),
        _rj(_fl(p_uc)  + _sig(p_uc),  WCOL[4]),
        _rj(_fl(lr_ind),         WCOL[5]),
        _rj(_fl(p_ind) + _sig(p_ind), WCOL[6]),
        _rj(_fl(lr_cc),          WCOL[7]),
        _rj(_fl(p_cc)  + _sig(p_cc),  WCOL[8]),
    ]
    print('    ' + SEP.join(row))


# ── Rolling 1-step OOS QLIKE（window WIN_OOS）───────────────────────────────
WIN_OOS        = min(WIN_OOS_MAX, T_pred // 4)
roll_qlike_oos = np.full(T_pred, np.nan)
for t in range(WIN_OOS, T_pred):
    sl = slice(t - WIN_OOS, t)
    h_sl  = h_roll[sl]
    rv_sl = rv_pred_m[sl]
    roll_qlike_oos[t] = float(np.nanmean(np.log(h_sl) + rv_sl / h_sl))

# ── 粒子 1-step 90% CI（向量化）─────────────────────────────────────────────
mu_all     = (x_oos_states
              + theta_last[None, :, 1] * (theta_last[None, :, 2] - x_oos_states))
h_roll_q05 = np.quantile(np.maximum(mu_all, 1e-10), 0.05, axis=1)
h_roll_q95 = np.quantile(np.maximum(mu_all, 1e-10), 0.95, axis=1)

# ── IS 滾動 QLIKE（window WIN_OOS）──────────────────────────────────────────
T_is          = T - 1
roll_qlike_is = np.full(T_is, np.nan)
for t in range(WIN_OOS, T_is):
    sl = slice(t - WIN_OOS, t)
    h_sl_is  = h_is[1:][sl]
    rv_sl_is = rv_is_m[sl]
    roll_qlike_is[t] = float(np.nanmean(np.log(h_sl_is) + rv_sl_is / h_sl_is))

# ── 圖：IS 1-step forecast vs z² / rolling QLIKE ────────────────────────────
t_is_arr      = np.arange(1, T)
_proxy_lbl_is = f'z² proxy (rolling {RV_WIN}d)' if RV_WIN > 1 else 'z² proxy (逐期，去漂移)'

fig_is, axes_is = plt.subplots(2, 1, figsize=(12, 8))
axes_is[0].plot(t_is_arr, rv_is_m,  label=_proxy_lbl_is,              alpha=0.6)
axes_is[0].plot(t_is_arr, h_is[1:], label='IS 1-step (posterior mean)', linewidth=1.5)
axes_is[0].set_title(f'IS 1-step Forecast vs z² Proxy - {LABEL}')
axes_is[0].legend()
axes_is[1].plot(np.arange(T_is), roll_qlike_is)
axes_is[1].set_title(f'IS Rolling QLIKE (W={WIN_OOS}) - {LABEL}')
axes_is[1].set_xlabel('t (IS)')
plt.suptitle('In-Sample Volatility Forecast & QLIKE', fontsize=13)
plt.tight_layout(); plt.show()

# ── 圖：OOS 滾動 1-step forecast vs realized RV / rolling QLIKE ──────────────
t_pred_arr = np.arange(T_pred)
_proxy_lbl = f'z² proxy (rolling {RV_WIN}d)' if RV_WIN > 1 else 'z² proxy (逐期，去漂移)'

fig_oos, axes_oos = plt.subplots(2, 1, figsize=(12, 8))
axes_oos[0].plot(t_pred_arr, rv_pred_m, label=_proxy_lbl,                        alpha=0.6)
axes_oos[0].plot(t_pred_arr, h_roll,    label='Rolling 1-step (posterior mean)', linewidth=1.5)
axes_oos[0].fill_between(t_pred_arr, h_roll_q05, h_roll_q95, alpha=0.2, label='90% CI')
axes_oos[0].set_title(f'OOS 1-step Forecast vs z² Proxy - {LABEL}')
axes_oos[0].legend()
axes_oos[1].plot(t_pred_arr, roll_qlike_oos)
axes_oos[1].set_title(f'OOS Rolling QLIKE (W={WIN_OOS}) - {LABEL}')
axes_oos[1].set_xlabel('t (OOS)')
plt.tight_layout(); plt.show()
