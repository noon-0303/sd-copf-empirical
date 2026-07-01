import sys
import unicodedata
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import warnings
import pyvinecopulib as pv

from Empirical.copulafitall_pd_2 import copulafitall23

warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\user\PycharmProjects\sd-copf')


###################  ( 0.1 )  — 資料載入  ############################################################

DATA_PATH = r"C:\Users\user\PycharmProjects\sd-copf\Empirical\data\real_data2-1.xlsx"
data1 = pd.read_excel(DATA_PATH, sheet_name='return1', header=None).values
data2 = pd.read_excel(DATA_PATH, sheet_name='return2', header=None).values

y1 = data1
y2 = data2
T, Y = y1.shape

var1_emp = np.var(y1, axis=0, ddof=1)
var2_emp = np.var(y2, axis=0, ddof=1)

N = 300
k = 5
p = 2
h = T * 1e-4
a = np.sqrt(1.0 - h**2)

copula_list = ["copulaN", "copulaT", "copulaC", "copulaF", "copulaG"]
n_cop = len(copula_list)

d = 1.0 / 10

KAPPA_SCALE = 1.0
SIGMA_SCALE = 1.0
MIN_VAR     = 1e-7

def _est_sv_params(y_s, v_s):
    mu_e  = float(np.mean(y_s))

    r     = y_s - mu_e
    proxy = r**2
    lam   = 0.94
    rv    = pd.Series(proxy).ewm(alpha=1 - lam).mean().values
    rv    = np.clip(rv, MIN_VAR, None)

    theta_e = float(max(np.mean(proxy), MIN_VAR))

    x_ar    = rv[:-1]
    yv      = rv[1:]
    b       = np.cov(x_ar, yv)[0, 1] / np.var(x_ar)
    kappa_e = float(np.clip(1.0 - b, 1e-3, 0.999))

    ar1_int = float(np.mean(yv) - b * np.mean(x_ar))
    u       = yv - ar1_int - b * x_ar
    sigma_e = float(np.sqrt(np.mean(u**2 / np.clip(x_ar, MIN_VAR, None))))
    sigma_e = max(sigma_e, 1e-4)

    eps1     = r[1:] / np.sqrt(np.clip(x_ar, MIN_VAR, None))
    eta      = u / (sigma_e * np.sqrt(np.clip(x_ar, MIN_VAR, None)))
    corr_lev = np.corrcoef(eps1, eta)[0, 1]
    rho_e    = float(np.clip(corr_lev if np.isfinite(corr_lev) else 0.0, -0.999, 0.999))

    kappa_e *= KAPPA_SCALE
    sigma_e  = max(sigma_e * SIGMA_SCALE, 1e-4)

    if 2 * kappa_e * theta_e < sigma_e**2:
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

R_mat          = np.zeros((Y, n_cop))
logLik         = np.zeros((Y, n_cop))
logLik_all     = np.zeros(Y)

iter_arr       = np.zeros((Y, n_cop), dtype=int)
theta1_estimate = np.zeros((Y, k, n_cop))
theta2_estimate = np.zeros((Y, k, n_cop))

ks_stat  = np.zeros((Y, 2))
ks_pval  = np.zeros((Y, 2))
lb_pval  = np.zeros((Y, 2))
lb2_pval = np.zeros((Y, 2))


# ──── 資料迴圈(最外圍)  ─────────────────────────────────────────────────────────────────────────────────


for j in range(Y):

    #########################   ( 0.2 ) 迴圈中變數   ##################################################

    y   = np.column_stack([y1[:, j], y2[:, j]])

    v1j = max(float(var1_emp[j]), MIN_VAR)
    v2j = max(float(var2_emp[j]), MIN_VAR)

    c1, v1_arr = _est_sv_params(y1[:, j], v1j)
    c2, v2_arr = _est_sv_params(y2[:, j], v2j)
    v_all1 = v1_arr**2 * d
    v_all2 = v2_arr**2 * d

    x      = np.zeros((T, N, p))
    mu     = np.zeros((T, N, p))
    theta1 = np.zeros((T, N, k))
    theta2 = np.zeros((T, N, k))
    m1     = np.zeros((T, N, k))
    m2     = np.zeros((T, N, k))

    theta1_s = np.zeros((T, k, n_cop))   # 各時點粒子平均參數（各 copula）
    theta2_s = np.zeros((T, k, n_cop))
    x_s      = np.zeros((T, p, n_cop))

    vv_1 = np.zeros((T, k))
    vv_2 = np.zeros((T, k))

    rho_Ns = np.zeros((T, n_cop))   # 各時點 Kendall's τ

    # 抽樣兩個市場的初始波動度、參數  ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆

    #    波動度    #
    x[0, :, 0] = np.random.normal(v1j, v1j*d, N)
    neg = x[0, :, 0] < 0
    x[0, neg, 0] = 2*v1j - x[0, neg, 0]

    x[0, :, 1] = np.random.normal(v2j, v2j*d, N)
    neg = x[0, :, 1] < 0
    x[0, neg, 1] = 2*v2j - x[0, neg, 1]

    mu[0, :, :] = x[0, :, :]
    x_s[0, :, :] = np.tile(np.mean(x[0, :, :], axis=0)[:, np.newaxis], (1, n_cop))

    #    參數    #
    theta1[0, :, 0] = np.random.normal(c1[0], v1_arr[0], N)

    theta1[0, :, 1] = np.clip(np.random.gamma(c1[1]**2/v1_arr[1]**2, v1_arr[1]**2/c1[1], N), 1e-6, np.inf)

    theta1[0, :, 2] = np.clip(np.random.gamma(v1j**2/v1_arr[2]**2, v1_arr[2]**2/v1j, N), 1e-6, np.inf)

    theta1[0, :, 3] = np.clip(np.random.gamma(c1[3]**2/v1_arr[3]**2, v1_arr[3]**2/c1[3], N), 1e-6, np.inf)

    _a, _b = c1[4] - np.sqrt(3)*v1_arr[4], c1[4] + np.sqrt(3)*v1_arr[4]
    theta1[0, :, 4] = np.clip(np.random.uniform(_a, _b, N), -0.999, 0.999)


    theta2[0, :, 0] = np.random.normal(c2[0], v2_arr[0], N)

    theta2[0, :, 1] = np.clip(np.random.gamma(c2[1]**2/v2_arr[1]**2, v2_arr[1]**2/c2[1], N), 1e-6, np.inf)

    theta2[0, :, 2] = np.clip(np.random.gamma(v2j**2/v2_arr[2]**2, v2_arr[2]**2/v2j, N), 1e-6, np.inf)

    theta2[0, :, 3] = np.clip(np.random.gamma(c2[3]**2/v2_arr[3]**2, v2_arr[3]**2/c2[3], N), 1e-6, np.inf)

    _a, _b = c2[4] - np.sqrt(3)*v2_arr[4], c2[4] + np.sqrt(3)*v2_arr[4]
    theta2[0, :, 4] = np.clip(np.random.uniform(_a, _b, N), -0.999, 0.999)

    theta1_s[0, :, :] = np.tile(np.mean(theta1[0, :, :], axis=0)[:, np.newaxis], (1, n_cop))
    theta2_s[0, :, :] = np.tile(np.mean(theta2[0, :, :], axis=0)[:, np.newaxis], (1, n_cop))



    #########################  ( 0.3 ) —  隨機變數  ############################################


    r2 = np.random.randn(T, N, p)

    x0_init      = x[0].copy()
    theta1_0init = theta1[0].copy()
    theta2_0init = theta2[0].copy()

    best_ci_so_far = 0
    x_best       = None
    theta1_best  = None
    theta2_best  = None
    theta1_final = None
    theta2_final = None
    x_final      = None



    # ── Copula 迴圈 ─────────────────────────────────────────────────────────────────────────────────


    for ci, c in enumerate(copula_list):

        # 每個 copula 從相同初始狀態出發
        x[0]      = x0_init.copy()
        theta1[0] = theta1_0init.copy()
        theta2[0] = theta2_0init.copy()
        mu[0]     = x0_init.copy()


        ###################### ( 0.4 ) — 初始權重 ####################################################


        w = np.zeros((T, N))
        w[0, :] = 1.0 / N


        # ── 時間迴圈 ────────────────────────────────────────────────────────────────────────────────


        for t in range(1, T):

            ################## ( 1.1 ) — 平滑參數（ resampling 前 ） ##################################

            w_prev = w[t - 1, :]
            theta_mean = np.sum(w_prev[:, np.newaxis] * theta1[t - 1, :, :], axis=0)
            m1[t - 1, :, :] = a * theta1[t - 1, :, :] + (1.0 - a) * theta_mean

            theta_mean = np.sum(w_prev[:, np.newaxis] * theta2[t - 1, :, :], axis=0)
            m2[t - 1, :, :] = a * theta2[t - 1, :, :] + (1.0 - a) * theta_mean


            ################## ( 1.2 ) — 預測 mu（ 下一期波動度 ） #####################################

            mu[t, :, 0] = (x[t - 1, :, 0]
                           + m1[t - 1, :, 1] * (m1[t - 1, :, 2] - x[t - 1, :, 0]))
            mu[t, :, 1] = (x[t - 1, :, 1]
                           + m2[t - 1, :, 1] * (m2[t - 1, :, 2] - x[t - 1, :, 1]))


            ################## ( 1.3 ) — copula likelihood g  #####################################

            MU    = (np.column_stack([m1[t - 1, :, 0], m2[t - 1, :, 0]])
                     - mu[t, :, :] / 2.0)                           # (N, 2)
            SIGMA = np.sqrt(np.maximum(mu[t, :, :], 1e-12))         # (N, 2)
            y_t   = np.tile(y[t, :], (N, 1))                        # (N, 2)
            u_t   = stats.norm.cdf(y_t, MU, SIGMA)
            y_p   = stats.norm.pdf(y_t, MU, SIGMA)

            cop_pdf, _ = copulafitall23(copula_list, c,
                                       np.column_stack([u_t[:, 0], u_t[:, 1]]))
            g = cop_pdf * y_p[:, 0] * y_p[:, 1]
            g = w[t - 1, :] * g
            g_sum = np.sum(g)
            if g_sum > 0:
                g = g / g_sum
                g = g / g.sum()   # 消除浮點漂移
            else:
                g = np.full(N, 1.0 / N)


            ################ ( 2.1 ) — 重抽樣參數 ##################################

            rs = np.random.choice(N, N, replace=True, p=g)
            x[t - 1, :, :]      = x[t - 1, rs, :]
            theta1[t - 1, :, :] = theta1[t - 1, rs, :]
            theta2[t - 1, :, :] = theta2[t - 1, rs, :]


            ################## ( 2.2 ) — 計算平滑參數、波動度（resampling 後） #############################

            theta_mean = np.mean(theta1[t - 1, :, :], axis=0)
            m1[t - 1, :, :] = a * theta1[t - 1, :, :] + (1.0 - a) * theta_mean
            v_1 = np.maximum(np.var(theta1[t - 1, :, :], axis=0, ddof=1), v_all1)

            theta_mean = np.mean(theta2[t - 1, :, :], axis=0)
            m2[t - 1, :, :] = a * theta2[t - 1, :, :] + (1.0 - a) * theta_mean
            v_2 = np.maximum(np.var(theta2[t - 1, :, :], axis=0, ddof=1), v_all2)

            vv_1[t, :] = v_1
            vv_2[t, :] = v_2

            mu[t, :, 0] = (x[t - 1, :, 0]
                           + m1[t - 1, :, 1] * (m1[t - 1, :, 2] - x[t - 1, :, 0]))
            mu[t, :, 1] = (x[t - 1, :, 1]
                           + m2[t - 1, :, 1] * (m2[t - 1, :, 2] - x[t - 1, :, 1]))
            mu[t, :, :] = np.abs(mu[t, :, :])   # 確保 sqrt 合法


            ################## ( 2.3 ) — 抽樣下一期參數 ##################################### ☆ ☆ ☆

            theta1[t, :, 0] = np.random.normal(m1[t - 1, :, 0], np.sqrt(h**2 * v_1[0]), N)
            theta2[t, :, 0] = np.random.normal(m2[t - 1, :, 0], np.sqrt(h**2 * v_2[0]), N)

            _sd1 = np.sqrt(h**2 * v_1[1])
            theta1[t, :, 1] = np.clip(np.random.gamma(m1[t-1,:,1]**2/_sd1**2, _sd1**2/m1[t-1,:,1]), 1e-6, np.inf)
            _sd2 = np.sqrt(h**2 * v_2[1])
            theta2[t, :, 1] = np.clip(np.random.gamma(m2[t-1,:,1]**2/_sd2**2, _sd2**2/m2[t-1,:,1]), 1e-6, np.inf)

            _sd1 = np.sqrt(h**2 * v_1[2])
            theta1[t, :, 2] = np.clip(np.random.gamma(m1[t-1,:,2]**2/_sd1**2, _sd1**2/m1[t-1,:,2]), 1e-6, np.inf)
            _sd2 = np.sqrt(h**2 * v_2[2])
            theta2[t, :, 2] = np.clip(np.random.gamma(m2[t-1,:,2]**2/_sd2**2, _sd2**2/m2[t-1,:,2]), 1e-6, np.inf)

            _sd1 = np.sqrt(h**2 * v_1[3])
            theta1[t, :, 3] = np.clip(np.random.gamma(m1[t-1,:,3]**2/_sd1**2, _sd1**2/m1[t-1,:,3]), 1e-6, np.inf)
            _sd2 = np.sqrt(h**2 * v_2[3])
            theta2[t, :, 3] = np.clip(np.random.gamma(m2[t-1,:,3]**2/_sd2**2, _sd2**2/m2[t-1,:,3]), 1e-6, np.inf)

            _sd1 = np.sqrt(h**2 * v_1[4])
            _a1 = m1[t-1, :, 4] - np.sqrt(3)*_sd1;  _b1 = m1[t-1, :, 4] + np.sqrt(3)*_sd1
            theta1[t, :, 4] = np.clip(np.random.uniform(_a1, _b1), -0.999, 0.999)
            _sd2 = np.sqrt(h**2 * v_2[4])
            _a2 = m2[t-1, :, 4] - np.sqrt(3)*_sd2;  _b2 = m2[t-1, :, 4] + np.sqrt(3)*_sd2
            theta2[t, :, 4] = np.clip(np.random.uniform(_a2, _b2), -0.999, 0.999)


            ################## ( 2.4 ) — 用新參數生成波動度 ###########################################

            x_prev0 = np.maximum(x[t - 1, :, 0], 1e-12)
            r1 = (y[t, 0] - (theta1[t, :, 0] - 0.5 * x_prev0)) / np.sqrt(x_prev0)
            x[t, :, 0] = (x_prev0
                          + theta1[t, :, 1] * (theta1[t, :, 2] - x_prev0)
                          + theta1[t, :, 3] * np.sqrt(x_prev0)
                          * (theta1[t, :, 4] * r1
                             + np.sqrt(np.maximum(1.0 - theta1[t, :, 4]**2, 0.0))
                             * r2[t, :, 0]))

            x_prev1 = np.maximum(x[t - 1, :, 1], 1e-12)
            r1 = (y[t, 1] - (theta2[t, :, 0] - 0.5 * x_prev1)) / np.sqrt(x_prev1)
            x[t, :, 1] = (x_prev1
                          + theta2[t, :, 1] * (theta2[t, :, 2] - x_prev1)
                          + theta2[t, :, 3] * np.sqrt(x_prev1)
                          * (theta2[t, :, 4] * r1
                             + np.sqrt(np.maximum(1.0 - theta2[t, :, 4]**2, 0.0))
                             * r2[t, :, 1]))

            negative_index = x[t, :, :] < 0
            x[t, :, :]    = np.abs(x[t, :, :])


            ################## ( 3.1 ) — 計算 copula ( 用來計算權重 ) ##############################

            MU    = (np.column_stack([theta1[t, :, 0], theta2[t, :, 0]])
                     - x[t, :, :] / 2.0)
            SIGMA = np.sqrt(np.maximum(x[t, :, :], 1e-12))
            y_t   = np.tile(y[t, :], (N, 1))
            u_t   = stats.norm.cdf(y_t, MU, SIGMA)
            y_p   = stats.norm.pdf(y_t, MU, SIGMA)

            cop_pdf, rho_hat = copulafitall23(
                copula_list, c, np.column_stack([u_t[:, 0], u_t[:, 1]]))
            rho_Ns[t, ci] = rho_hat
            lik_new = cop_pdf * y_p[:, 0] * y_p[:, 1]

            MU    = (np.column_stack([m1[t - 1, :, 0], m2[t - 1, :, 0]])
                     - mu[t, :, :] / 2.0)
            SIGMA = np.sqrt(np.maximum(mu[t, :, :], 1e-12))
            u_t   = stats.norm.cdf(y_t, MU, SIGMA)
            y_p   = stats.norm.pdf(y_t, MU, SIGMA)

            cop_pdf2, _ = copulafitall23(
                copula_list, c, np.column_stack([u_t[:, 0], u_t[:, 1]]))
            lik_pred = cop_pdf2 * y_p[:, 0] * y_p[:, 1]

            w[t, :] = lik_new / np.maximum(lik_pred, 1e-300)
            w[t, negative_index[:, 0]] = 0.0
            w[t, negative_index[:, 1]] = 0.0
            w_sum = np.sum(w[t, :])
            w[t, :] = w[t, :] / w_sum if w_sum > 0 else np.full(N, 1.0 / N)

            theta1_s[t, :, ci] = np.mean(theta1[t, :, :], axis=0)
            theta2_s[t, :, ci] = np.mean(theta2[t, :, :], axis=0)
            x_s[t, :, ci]      = np.mean(x[t, :, :], axis=0)


        # ── logLik (still in copula loop) ──────────────────────────────────────────────────────────

        iter_arr[j, ci] = T - 1

        MU_ll  = (np.column_stack([theta1_s[:, 0, ci], theta2_s[:, 0, ci]])
                  - x_s[:, :, ci] / 2.0)                            # (T, 2)
        SIG_ll = np.sqrt(np.maximum(x_s[:, :, ci], 1e-12))          # (T, 2)
        u_ll   = np.clip(stats.norm.cdf(y, MU_ll, SIG_ll), 1e-4, 1 - 1e-4)
        yp_ll  = stats.norm.pdf(y, MU_ll, SIG_ll)
        cop_ll, rho_ll = copulafitall23(copula_list, c, u_ll)
        jpdf   = np.maximum(cop_ll * yp_ll[:, 0] * yp_ll[:, 1], np.finfo(float).tiny)
        logLik[j, ci]     = np.sum(np.log(jpdf))

        if x_best is None or logLik[j, ci] > logLik[j, best_ci_so_far]:
            best_ci_so_far = ci
            x_best       = x_s[:, :, ci].copy()
            theta1_best  = theta1_s[:, :, ci].copy()
            theta2_best  = theta2_s[:, :, ci].copy()
            theta1_final = theta1[T-1].copy()   # (N, k) 最佳 copula t=T-1 粒子後驗
            theta2_final = theta2[T-1].copy()
            x_final      = x[T-1].copy()        # (N, 2) 最佳 copula t=T-1 狀態後驗

        R_mat[j, ci] = rho_ll
        theta1_estimate[j, :, ci] = theta1_s[T - 1, :, ci]
        theta2_estimate[j, :, ci] = theta2_s[T - 1, :, ci]



   #-----------end of copula loop------------------------------------------------------------


    # ── 選最佳 copula（logLik 最大）───────────

    best_ci = int(np.argmax(logLik[j]))
    logLik_all[j] = logLik[j, best_ci]


    # ── LB test（以最佳 copula 收斂粒子） ──────
    mu_hat  = np.column_stack([
        theta1_best[:, 0] - x_best[:, 0] / 2,
        theta2_best[:, 0] - x_best[:, 1] / 2
    ])
    sig_hat = np.sqrt(np.maximum(x_best, 1e-10))
    z = (y - mu_hat) / sig_hat
    u = stats.norm.cdf(z)

    for asset in range(2):
        ks_res            = stats.kstest(u[:, asset], 'uniform')
        ks_stat[j, asset] = ks_res.statistic
        ks_pval[j, asset] = ks_res.pvalue
        lb_pval[j,  asset] = acorr_ljungbox(z[:, asset],    lags=[20], return_df=True)['lb_pvalue'].iloc[-1]
        lb2_pval[j, asset] = acorr_ljungbox(z[:, asset]**2, lags=[20], return_df=True)['lb_pvalue'].iloc[-1]


    print(f'one data loop  j={j+1}/{Y}')




# ── data loop end ──────────────────────────────────────────────────────────────────────────────


t_arr = np.arange(T)

fig, axes = plt.subplots(2, 2, figsize=(14, 6))
axes[0, 0].plot(t_arr, y[:, 0])
axes[0, 0].set_title('Returns - 資產1')
axes[0, 1].plot(t_arr, y[:, 1])
axes[0, 1].set_title('Returns - 資產2')
axes[1, 0].plot(t_arr, x_best[:, 0], label='Filtered')
axes[1, 0].set_title('Volatility - 資產1')
axes[1, 0].legend()
axes[1, 1].plot(t_arr, x_best[:, 1], label='Filtered')
axes[1, 1].set_title('Volatility - 資產2')
axes[1, 1].legend()
plt.tight_layout(); plt.show()


# ── 各 Copula Kendall's τ ────────────────────────────────────────────────────
print('\n── 各 Copula Kendall\'s τ ──')
for ci, name in enumerate(copula_list):
    print(f'  {name}: τ = {R_mat[0, ci]:.4f}')

# ── logLik 與最佳 Copula ──────────────────────────────────────────────────────
print('\n── 各 Copula logLik ──')
for ci, name in enumerate(copula_list):
    print(f'  {name}: {logLik[0, ci]:.4f}')

best_ci_out = best_ci_so_far   # 與 391 行 theta1_final/x_final 的選取依據一致
print(f'\n最佳 Copula: {copula_list[best_ci_out]}  (logLik = {np.max(logLik[0]):.4f})')

# ── 殘差診斷（最佳 Copula）────────────────────────────────────────────────────
print(f'\n── 殘差診斷（最佳 Copula: {copula_list[best_ci_out]}）──')
asset_labels = ['資產1', '資產2']
for asset_idx, aname in enumerate(asset_labels):
    print(f'\n  {aname}:')
    print(f'    KS  (PIT vs U[0,1]):  stat={ks_stat[0, asset_idx]:.4f}  p={ks_pval[0, asset_idx]:.4f}{"  （拒絕）" if ks_pval[0, asset_idx] < 0.05 else "  （未拒絕）"}')
    print(f'    LB(20)  殘差自相關:    p={lb_pval[0, asset_idx]:.4f}{"  （拒絕）" if lb_pval[0, asset_idx] < 0.05 else "  （未拒絕）"}')
    print(f'    LB²(20) ARCH 效果:     p={lb2_pval[0, asset_idx]:.4f}{"  （拒絕）" if lb2_pval[0, asset_idx] < 0.05 else "  （未拒絕）"}')

# ── 最佳 Copula 估計參數 ──────────────────────────────────────────────────────
param_labels = ['mu', 'kappa', 'theta', 'sigma', 'rho']
print('\n── 最佳 Copula 估計參數 ──')
print(f'    {"":8s}  {"assets1":>12s}  {"assets2":>12s}')
for pi, pn in enumerate(param_labels):
    print(f'    {pn:8s}: {theta1_estimate[0, pi, best_ci_out]:12.6f}  {theta2_estimate[0, pi, best_ci_out]:12.6f}')



# ─────────────────────────────────────────────────────────────────────────────
# ── VaR / ES 計算 ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


VaR_LEVELS = [0.01, 0.05]

# ── A. 個別資產滾動條件 VaR / ES（實證分布）────────────────────────
print('\n── 個別資產條件 VaR / ES（實證分布）──')
hdr = f'    {"":10s}  {"資產1"+" 均值":>12s}  {"資產2"+" 均值":>12s}'
print(hdr)
for alpha in VaR_LEVELS:
    z_alpha  = np.quantile(z, alpha, axis=0)                        # (2,) 各資產實證分位數
    cond_VaR = -(mu_hat + z_alpha * sig_hat)                        # (T, 2) 正值=損失
    cond_ES  = np.column_stack([
        -(mu_hat[:, ai] + np.mean(z[z[:, ai] <= z_alpha[ai], ai]) * sig_hat[:, ai])
        for ai in range(2)
    ])                                                               # (T, 2)
    pct = int((1 - alpha) * 100)
    print(f'    VaR {pct}%:  {cond_VaR[:, 0].mean():10.6f}  {cond_VaR[:, 1].mean():10.6f}')
    print(f'    ES  {pct}%:  {cond_ES[:, 0].mean():10.6f}  {cond_ES[:, 1].mean():10.6f}')


# ── B. VaR 回測：Kupiec POF & Christoffersen CC 檢定 ──────────────────────────
def _kupiec_pof(hit, alpha):
    """Kupiec (1995) POF：無條件涵蓋，LR_uc ~ χ²(1)。"""
    T_, n_viol = len(hit), int(hit.sum())
    p_hat = n_viol / T_
    if n_viol == 0 or n_viol == T_:
        return np.nan, np.nan
    lr = -2.0 * (n_viol * np.log(alpha / p_hat) + (T_ - n_viol) * np.log((1 - alpha) / (1 - p_hat)))
    return lr, float(1.0 - stats.chi2.cdf(lr, df=1))


def _christoffersen_ind(hit):
    """Christoffersen (1998) 獨立性，LR_ind ~ χ²(1)。"""
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


def _dw(s):
    """字串顯示寬度（CJK 全形字元算 2，其餘算 1）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in str(s))

def _lj(s, w): return str(s) + ' ' * max(0, w - _dw(str(s)))
def _rj(s, w): return ' ' * max(0, w - _dw(str(s))) + str(s)
def _fl(v):    return f'{v:.4f}' if not np.isnan(v) else 'nan'
def _sig(p):   return '*' if (not np.isnan(p) and p < 0.05) else ' '

WCOL = [  14,     9,     7,      8,     7,      8,     7,     8,    7]
HCOL = ['資產', '違反/T', '違反率',
        'LR_POF', 'p_POF', 'LR_ind', 'p_ind', 'LR_cc', 'p_cc']



# ─────────────────────────────────────────────────────────────────────────────
# ── SV 參數有效性檢定（κ > 0, θ > 0, σ > 0）────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#
# def _ttest_gt0(arr):
#     """單側 t 檢定：H₀: μ ≤ 0，H₁: μ > 0。回傳 (均值, t, p)。"""
#     t_stat, p = stats.ttest_1samp(arr, popmean=0, alternative='greater')
#     return float(np.mean(arr)), float(t_stat), float(p)
#
# PIDX = [1,         2,         3        ]
# PLAB = ['κ > 0', 'θ > 0', 'σ > 0']
# PDSC = ['均值回歸為正', '長期變異數為正', '波動率為正']
#
# PW   = [16,        10,     8,      7]
#
# print(f'\n── SV 參數有效性檢定（最佳 Copula: {copula_list[best_ci_out]}）──')
# print('    H₀: 各參數 ≤ 0；單側 t 檢定，* p < 0.05 拒絕 H₀（確認條件成立）')
# print(f'\n    {"":16s}  {"─── 資產1 ───":^27s}  {"─── 資產2 ───":^27s}')
# print('    ' + '  '.join([
#     _rj('檢定', PW[0]), _rj('估計值', PW[1]), _rj('t-stat', PW[2]), _rj('p 值', PW[3]),
#     _rj('估計值', PW[1]), _rj('t-stat', PW[2]), _rj('p 值', PW[3]),
# ]))
# print('    ' + '  '.join('-' * w for w in [PW[0]] + PW[1:] * 2))
#
# for pidx, plab, pdsc in zip(PIDX, PLAB, PDSC):
#     m1_, t1, p1 = _ttest_gt0(theta1_best[:, pidx])
#     m2_, t2, p2 = _ttest_gt0(theta2_best[:, pidx])
#     label = f'{plab}（{pdsc}）'
#     row = [
#         _lj(label,           PW[0]),
#         _rj(f'{m1_:.6f}',    PW[1]),
#         _rj(f'{t1:.4f}',     PW[2]),
#         _rj(f'{p1:.4f}' + _sig(p1), PW[3]),
#         _rj(f'{m2_:.6f}',    PW[1]),
#         _rj(f'{t2:.4f}',     PW[2]),
#         _rj(f'{p2:.4f}' + _sig(p2), PW[3]),
#     ]
#     print('    ' + '  '.join(row))
#
# # Feller 條件：2κθ − σ² > 0
# feller1 = 2 * theta1_best[:, 1] * theta1_best[:, 2] - theta1_best[:, 3]**2
# feller2 = 2 * theta2_best[:, 1] * theta2_best[:, 2] - theta2_best[:, 3]**2
# mf1, tf1, pf1 = _ttest_gt0(feller1)
# mf2, tf2, pf2 = _ttest_gt0(feller2)
# feller_row = [
#     _lj('2κθ > σ²（Feller 條件）', PW[0]),
#     _rj(f'{mf1:.6f}',  PW[1]),
#     _rj(f'{tf1:.4f}',  PW[2]),
#     _rj(f'{pf1:.4f}' + _sig(pf1), PW[3]),
#     _rj(f'{mf2:.6f}',  PW[1]),
#     _rj(f'{tf2:.4f}',  PW[2]),
#     _rj(f'{pf2:.4f}' + _sig(pf2), PW[3]),
# ]
# print('    ' + '  '.join(feller_row))
#

# ─────────────────────────────────────────────────────────────────────────────
# ── Volatility Forecasting & QLIKE ───────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

FCST_HORIZONS = [1, 5, 10, 22]
WIN_OOS_MAX   = 120
RV_WIN        = 10


def _dm_test(loss1, loss2, h_horizon=1):
    """
    Diebold-Mariano (1995) + Harvey-Leybourne-Newbold (1997) 小樣本修正。
    H₀: E[loss1 − loss2] = 0
    Returns: DM stat, p-value (two-sided), p-value (one-sided: loss1 < loss2)
    """
    d      = loss1 - loss2
    d      = d[~np.isnan(d)]
    n      = len(d)
    if n < 2:
        return np.nan, np.nan, np.nan
    d_bar  = np.mean(d)
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv    = gamma0
    for lag in range(1, h_horizon):
        w_lag   = 1.0 - lag / h_horizon
        gamma_l = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        lrv    += 2.0 * w_lag * gamma_l
    lrv    = max(lrv, 1e-20)
    dm     = d_bar / np.sqrt(lrv / n)
    p_two  = 2.0 * float(stats.t.sf(abs(dm), df=n - 1))
    p_less = float(stats.t.cdf(dm, df=n - 1))
    return float(dm), p_two, p_less


PRED_PATH = r"C:\Users\user\PycharmProjects\sd-copf\Empirical\data\real_data2-1_pred.xlsx"
y_pred1 = pd.read_excel(PRED_PATH, sheet_name='return1', header=None).values.ravel()
y_pred2 = pd.read_excel(PRED_PATH, sheet_name='return2', header=None).values.ravel()
y_pred  = np.column_stack([y_pred1, y_pred2])
T_pred  = len(y_pred1)

# 逐期（t-1 資訊）κ／θ／μ，取代原本的全樣本時間平均常數
# （原本 np.mean(theta1_best[:, idx]) 會把 T-1 期「已收斂」的參數混進第 1 期的預測，
#   造成 IS forecast 對早期樣本有前視污染）
kappa1_hat_t = np.clip(theta1_best[:-1, 1], 1e-8, 10.0)   # (T-1,)
long1_hat_t  = theta1_best[:-1, 2]                         # (T-1,)
kappa2_hat_t = np.clip(theta2_best[:-1, 1], 1e-8, 10.0)
long2_hat_t  = theta2_best[:-1, 2]

# ── A. 樣本內 1-step-ahead forecast（用於 IS QLIKE 對照）─────────────────────
h_is    = np.empty((T, 2))
h_is[0] = x_best[0]
h_is[1:, 0] = x_best[:-1, 0] + kappa1_hat_t * (long1_hat_t - x_best[:-1, 0])
h_is[1:, 1] = x_best[:-1, 1] + kappa2_hat_t * (long2_hat_t - x_best[:-1, 1])
h_is = np.maximum(h_is, 1e-10)

mu1_is_t = theta1_best[:-1, 0]   # (T-1,)，逐期 μ（t-1 資訊）
mu2_is_t = theta2_best[:-1, 0]
drift_is = np.column_stack([
    mu1_is_t - 0.5 * x_best[:-1, 0],
    mu2_is_t - 0.5 * x_best[:-1, 1]
])
z_is    = y[1:] - drift_is
rv_is_m = np.column_stack([
    pd.Series(z_is[:, 0]**2).rolling(RV_WIN).mean().values,
    pd.Series(z_is[:, 1]**2).rolling(RV_WIN).mean().values
])

qlike_is   = np.nanmean(np.log(h_is[1:]) + rv_is_m / h_is[1:], axis=0)
h_const_is = np.nanmean(rv_is_m, axis=0)

# ── B. 滾動 OOS 序列粒子濾波（固定參數，逐期以觀測值更新波動度狀態）────────
r2_oos       = np.random.randn(T_pred, N, 2)
x_oos        = x_final.copy()
x_oos_states = np.empty((T_pred, N, 2))
h_roll       = np.empty((T_pred, 2))

for t in range(T_pred):
    x_oos_states[t] = x_oos

    mu1 = x_oos[:, 0] + theta1_final[:, 1] * (theta1_final[:, 2] - x_oos[:, 0])
    mu2 = x_oos[:, 1] + theta2_final[:, 1] * (theta2_final[:, 2] - x_oos[:, 1])
    h_roll[t, 0] = float(np.mean(np.maximum(mu1, 1e-10)))
    h_roll[t, 1] = float(np.mean(np.maximum(mu2, 1e-10)))

    if t < T_pred - 1:
        sx1  = np.sqrt(np.maximum(x_oos[:, 0], 1e-8))
        sx2  = np.sqrt(np.maximum(x_oos[:, 1], 1e-8))
        r1a  = (y_pred[t, 0] - (theta1_final[:, 0] - 0.5 * x_oos[:, 0])) / sx1
        r1b  = (y_pred[t, 1] - (theta2_final[:, 0] - 0.5 * x_oos[:, 1])) / sx2
        rho1 = theta1_final[:, 4]
        rho2 = theta2_final[:, 4]
        x_oos[:, 0] = np.maximum(
            x_oos[:, 0] + theta1_final[:, 1] * (theta1_final[:, 2] - x_oos[:, 0])
            + theta1_final[:, 3] * sx1
              * (rho1 * r1a + np.sqrt(np.maximum(1 - rho1**2, 0)) * r2_oos[t, :, 0]),
            1e-10)
        x_oos[:, 1] = np.maximum(
            x_oos[:, 1] + theta2_final[:, 1] * (theta2_final[:, 2] - x_oos[:, 1])
            + theta2_final[:, 3] * sx2
              * (rho2 * r1b + np.sqrt(np.maximum(1 - rho2**2, 0)) * r2_oos[t, :, 1]),
            1e-10)

mu1_oos   = float(np.mean(theta1_final[:, 0]))
mu2_oos   = float(np.mean(theta2_final[:, 0]))
drift_oos = np.column_stack([
    mu1_oos - 0.5 * np.mean(x_oos_states[:, :, 0], axis=1),
    mu2_oos - 0.5 * np.mean(x_oos_states[:, :, 1], axis=1)
])
z_pred    = y_pred - drift_oos
rv_pred_m = np.column_stack([
    pd.Series(z_pred[:, 0]**2).rolling(RV_WIN).mean().values,
    pd.Series(z_pred[:, 1]**2).rolling(RV_WIN).mean().values
])

# ── C. 樣本內 vs 樣本外 QLIKE 比較表 ─────────────────────────────────────────
qlike_oos       = np.nanmean(np.log(h_roll)    + rv_pred_m / h_roll,    axis=0)
qlike_bench_is  = np.log(h_const_is) + np.nanmean(rv_is_m   / h_const_is, axis=0)
qlike_bench_oos = np.log(h_const_is) + np.nanmean(rv_pred_m / h_const_is, axis=0)

print(f'\n── QLIKE 比較（realized proxy = {RV_WIN}-day rolling z²（去漂移殘差））──')
print(f'  {"":22s}  {"資產1":>12s}  {"資產2":>12s}')
print('  ' + '-' * 52)
print(f'  {"IS  SV-Copula":22s}  {qlike_is[0]:12.6f}  {qlike_is[1]:12.6f}')
print(f'  {"IS  常數基準":22s}  {qlike_bench_is[0]:12.6f}  {qlike_bench_is[1]:12.6f}')
print(f'  {"IS  差值(基準-模型)":22s}  {qlike_bench_is[0]-qlike_is[0]:12.6f}  {qlike_bench_is[1]-qlike_is[1]:12.6f}')
print('  ' + '-' * 52)
print(f'  {"OOS SV-Copula":22s}  {qlike_oos[0]:12.6f}  {qlike_oos[1]:12.6f}')
print(f'  {"OOS 常數基準(IS σ²)":22s}  {qlike_bench_oos[0]:12.6f}  {qlike_bench_oos[1]:12.6f}')
print(f'  {"OOS 差值(基準-模型)":22s}  {qlike_bench_oos[0]-qlike_oos[0]:12.6f}  {qlike_bench_oos[1]-qlike_oos[1]:12.6f}')
print('  （差值 > 0 表示模型優於常數基準）')

# ── D. Diebold-Mariano 檢定（1-step IS & OOS）───────────────────────────────
loss_sv_is     = np.log(h_is[1:])   + rv_is_m   / h_is[1:]
loss_bench_is  = np.log(h_const_is) + rv_is_m   / h_const_is
loss_sv_oos    = np.log(h_roll)     + rv_pred_m  / h_roll
loss_bench_oos = np.log(h_const_is) + rv_pred_m  / h_const_is

dm_is_1,  p2_is_1,  pl_is_1  = _dm_test(loss_sv_is[:, 0],  loss_bench_is[:, 0],  h_horizon=1)
dm_is_2,  p2_is_2,  pl_is_2  = _dm_test(loss_sv_is[:, 1],  loss_bench_is[:, 1],  h_horizon=1)
dm_oos_1, p2_oos_1, pl_oos_1 = _dm_test(loss_sv_oos[:, 0], loss_bench_oos[:, 0], h_horizon=1)
dm_oos_2, p2_oos_2, pl_oos_2 = _dm_test(loss_sv_oos[:, 1], loss_bench_oos[:, 1], h_horizon=1)

print('\n── Diebold-Mariano 檢定（SV-Copula vs 常數基準，QLIKE 損失差）──')
print('  H₀: 等預測精準度；H₁(less): SV 損失 < 基準損失（* p<0.05）')
print(f'  {"":12s}  {"─── 資產1 ───":^32s}  {"─── 資產2 ───":^32s}')
print(f'  {"":12s}  {"DM-stat":>10s}  {"p(two)":>8s}  {"p(less)":>9s}  {"DM-stat":>10s}  {"p(two)":>8s}  {"p(less)":>9s}')
print('  ' + '-' * 82)
print(f'  {"IS  (h=1)":12s}  {dm_is_1:10.4f}  {p2_is_1:8.4f}  {pl_is_1:8.4f}{_sig(pl_is_1)}  {dm_is_2:10.4f}  {p2_is_2:8.4f}  {pl_is_2:8.4f}{_sig(pl_is_2)}')
print(f'  {"OOS (h=1)":12s}  {dm_oos_1:10.4f}  {p2_oos_1:8.4f}  {pl_oos_1:8.4f}{_sig(pl_oos_1)}  {dm_oos_2:10.4f}  {p2_oos_2:8.4f}  {pl_oos_2:8.4f}{_sig(pl_oos_2)}')

# ─────────────────────────────────────────────────────────────────────────────
# ── Rolling h-step Forecast QLIKE & DM 檢定（樣本外）─────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

print('\n── Rolling h-step Forecast QLIKE & DM 檢定（樣本外，CIR 解析解）──')
print(f'  {"h":>4s}  {"Q1":>12s}  {"DM1":>9s}  {"p2_1":>7s}  {"pl_1":>7s}'
      f'  {"Q2":>12s}  {"DM2":>9s}  {"p2_2":>7s}  {"pl_2":>7s}  {"n":>6s}')
print('  ' + '-' * 100)

for h_fc in FCST_HORIZONS:
    n_periods = T_pred - h_fc + 1
    if n_periods <= 0:
        continue

    st   = x_oos_states[:n_periods]
    k1   = np.clip(theta1_final[None, :, 1], 0.0, 1.0)
    th1  = theta1_final[None, :, 2]
    k2   = np.clip(theta2_final[None, :, 1], 0.0, 1.0)
    th2  = theta2_final[None, :, 2]

    fcast1 = th1 + (st[:, :, 0] - th1) * (1 - k1)**h_fc
    fcast2 = th2 + (st[:, :, 1] - th2) * (1 - k2)**h_fc
    fh1    = np.mean(np.maximum(fcast1, 1e-10), axis=1)
    fh2    = np.mean(np.maximum(fcast2, 1e-10), axis=1)

    rv_tgt      = rv_pred_m[h_fc-1:h_fc-1+n_periods]
    loss_sv_h1  = np.log(fh1) + rv_tgt[:, 0] / fh1
    loss_sv_h2  = np.log(fh2) + rv_tgt[:, 1] / fh2
    loss_bch_h1 = np.log(h_const_is[0]) + rv_tgt[:, 0] / h_const_is[0]
    loss_bch_h2 = np.log(h_const_is[1]) + rv_tgt[:, 1] / h_const_is[1]

    q1 = float(np.nanmean(loss_sv_h1))
    q2 = float(np.nanmean(loss_sv_h2))
    dm_h1, p2_h1, pl_h1 = _dm_test(loss_sv_h1, loss_bch_h1, h_horizon=h_fc)
    dm_h2, p2_h2, pl_h2 = _dm_test(loss_sv_h2, loss_bch_h2, h_horizon=h_fc)
    print(f'  {h_fc:>4d}  {q1:12.6f}  {dm_h1:9.4f}  {p2_h1:7.4f}  {pl_h1:7.4f}{_sig(pl_h1)}'
          f'  {q2:12.6f}  {dm_h2:9.4f}  {p2_h2:7.4f}  {pl_h2:7.4f}{_sig(pl_h2)}  {n_periods:>6d}')

# ─────────────────────────────────────────────────────────────────────────────
# ── VaR 回測（OOS）：Kupiec POF & Christoffersen CC ──────────────────────────
# z_alpha 由 IS 殘差估計（無前視），條件 VaR 來自 OOS 粒子濾波 1-step 預測
# ─────────────────────────────────────────────────────────────────────────────

SEP         = '  '
mu_hat_oos  = drift_oos
sig_hat_oos = np.sqrt(np.maximum(h_roll, 1e-10))

for alpha in VaR_LEVELS:
    pct          = int((1 - alpha) * 100)
    z_alpha_is   = np.quantile(z, alpha, axis=0)
    cond_VaR_oos = -(mu_hat_oos + z_alpha_is * sig_hat_oos)

    series_oos = [
        ('資產1', y_pred[:, 0], cond_VaR_oos[:, 0]),
        ('資產2', y_pred[:, 1], cond_VaR_oos[:, 1]),
    ]
    print(f'\n── OOS VaR 回測 {pct}%（α={alpha:.2f}）—— Kupiec POF & Christoffersen CC ──')
    print('    (* p<0.05 拒絕 H₀)')
    print('    ' + SEP.join(_rj(hh, ww) for hh, ww in zip(HCOL, WCOL)))
    print('    ' + SEP.join('-' * ww for ww in WCOL))

    for lab, ret, var_ in series_oos:
        hit_oos       = ret < -var_
        n_viol        = int(hit_oos.sum())
        p_hat         = n_viol / T_pred
        lr_uc, p_uc   = _kupiec_pof(hit_oos, alpha)
        lr_ind, p_ind = _christoffersen_ind(hit_oos)
        lr_cc = (lr_uc + lr_ind) if not (np.isnan(lr_uc) or np.isnan(lr_ind)) else np.nan
        p_cc  = float(1.0 - stats.chi2.cdf(lr_cc, df=2)) if not np.isnan(lr_cc) else np.nan

        row = [
            _lj(lab,                       WCOL[0]),
            _rj(f'{n_viol}/{T_pred}',      WCOL[1]),
            _rj(f'{p_hat:.4f}',            WCOL[2]),
            _rj(_fl(lr_uc),                WCOL[3]),
            _rj(_fl(p_uc)  + _sig(p_uc),   WCOL[4]),
            _rj(_fl(lr_ind),               WCOL[5]),
            _rj(_fl(p_ind) + _sig(p_ind),  WCOL[6]),
            _rj(_fl(lr_cc),                WCOL[7]),
            _rj(_fl(p_cc)  + _sig(p_cc),   WCOL[8]),
        ]
        print('    ' + SEP.join(row))


# ── Rolling 1-step QLIKE（樣本外，window WIN_OOS）────────────────────────────
WIN_OOS        = min(WIN_OOS_MAX, T_pred // 4)
roll_qlike_oos = np.full((T_pred, 2), np.nan)
for t in range(WIN_OOS, T_pred):
    sl = slice(t - WIN_OOS, t)
    h_sl  = h_roll[sl]
    rv_sl = rv_pred_m[sl]
    roll_qlike_oos[t, 0] = float(np.nanmean(np.log(h_sl[:, 0]) + rv_sl[:, 0] / h_sl[:, 0]))
    roll_qlike_oos[t, 1] = float(np.nanmean(np.log(h_sl[:, 1]) + rv_sl[:, 1] / h_sl[:, 1]))

# ── 粒子 1-step 90% CI（向量化）─────────────────────────────────────────────
mu_all_1 = (x_oos_states[:, :, 0]
            + theta1_final[None, :, 1] * (theta1_final[None, :, 2] - x_oos_states[:, :, 0]))
mu_all_2 = (x_oos_states[:, :, 1]
            + theta2_final[None, :, 1] * (theta2_final[None, :, 2] - x_oos_states[:, :, 1]))
h_roll_q05 = np.column_stack([np.quantile(np.maximum(mu_all_1, 1e-10), 0.05, axis=1),
                               np.quantile(np.maximum(mu_all_2, 1e-10), 0.05, axis=1)])
h_roll_q95 = np.column_stack([np.quantile(np.maximum(mu_all_1, 1e-10), 0.95, axis=1),
                               np.quantile(np.maximum(mu_all_2, 1e-10), 0.95, axis=1)])

# ── IS 滾動 QLIKE（window WIN_OOS）──────────────────────────────────────────
T_is          = T - 1
roll_qlike_is = np.full((T_is, 2), np.nan)
for t in range(WIN_OOS, T_is):
    sl = slice(t - WIN_OOS, t)
    h_sl_is  = h_is[1:][sl]
    rv_sl_is = rv_is_m[sl]
    roll_qlike_is[t, 0] = float(np.nanmean(np.log(h_sl_is[:, 0]) + rv_sl_is[:, 0] / h_sl_is[:, 0]))
    roll_qlike_is[t, 1] = float(np.nanmean(np.log(h_sl_is[:, 1]) + rv_sl_is[:, 1] / h_sl_is[:, 1]))

# ── 圖：IS 1-step forecast vs z² / rolling QLIKE ────────────────────────────
t_is_arr      = np.arange(1, T)
_proxy_lbl_is = f'z² proxy (rolling {RV_WIN}d)' if RV_WIN > 1 else 'z² proxy (逐期，去漂移)'

fig_is, axes_is = plt.subplots(2, 2, figsize=(14, 8))
for ai, aname in enumerate(['資產1', '資產2']):
    axes_is[0, ai].plot(t_is_arr, rv_is_m[:, ai],  label=_proxy_lbl_is,              alpha=0.6)
    axes_is[0, ai].plot(t_is_arr, h_is[1:, ai],    label='IS 1-step (posterior mean)', linewidth=1.5)
    axes_is[0, ai].set_title(f'IS 1-step Forecast vs z² Proxy - {aname}')
    axes_is[0, ai].legend()
    axes_is[1, ai].plot(np.arange(T_is), roll_qlike_is[:, ai])
    axes_is[1, ai].set_title(f'IS Rolling QLIKE (W={WIN_OOS}) - {aname}')
    axes_is[1, ai].set_xlabel('t (IS)')
plt.suptitle('In-Sample Volatility Forecast & QLIKE', fontsize=13)
plt.tight_layout(); plt.show()

# ── 圖：OOS 滾動 1-step forecast vs realized RV / rolling QLIKE ──────────────
t_pred_arr = np.arange(T_pred)
_proxy_lbl = f'z² proxy (rolling {RV_WIN}d)' if RV_WIN > 1 else 'z² proxy (逐期，去漂移)'

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8))
for ai, aname in enumerate(['資產1', '資產2']):
    axes3[0, ai].plot(t_pred_arr, rv_pred_m[:, ai],  label=_proxy_lbl,                        alpha=0.6)
    axes3[0, ai].plot(t_pred_arr, h_roll[:, ai],      label='Rolling 1-step (posterior mean)', linewidth=1.5)
    axes3[0, ai].fill_between(t_pred_arr,
                               h_roll_q05[:, ai], h_roll_q95[:, ai],
                               alpha=0.2, label='90% CI')
    axes3[0, ai].set_title(f'OOS 1-step Forecast vs z² Proxy - {aname}')
    axes3[0, ai].legend()
    axes3[1, ai].plot(t_pred_arr, roll_qlike_oos[:, ai])
    axes3[1, ai].set_title(f'OOS Rolling QLIKE (W={WIN_OOS}) - {aname}')
    axes3[1, ai].set_xlabel('t (OOS)')
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ── (1) Rosenblatt OOS PIT：i.i.d. U(0,1) 檢定（無 look-ahead）─────────────
# ─────────────────────────────────────────────────────────────────────────────

# OOS 邊際 PIT（h_roll[t] 與 drift_oos[t] 均基於觀察 y_pred[t] 前的粒子狀態）
sig_oos_pit = np.sqrt(np.maximum(h_roll, 1e-10))          # (T_pred, 2)
u_oos_pit   = stats.norm.cdf((y_pred - drift_oos) / sig_oos_pit)
u_oos_pit   = np.clip(u_oos_pit, 1e-10, 1 - 1e-10)

# IS copula 固定擬合（只用 IS 邊際 PIT u，OOS 評估時無前視）
_FAMILY_MAP_PD = {
    'copulaN': pv.BicopFamily.gaussian,
    'copulaT': pv.BicopFamily.student,
    'copulaC': pv.BicopFamily.clayton,
    'copulaF': pv.BicopFamily.frank,
    'copulaG': pv.BicopFamily.gumbel,
}
bc_is = pv.Bicop(family=_FAMILY_MAP_PD[copula_list[best_ci_out]])
bc_is.fit(np.clip(u, 1e-6, 1 - 1e-6))

# 聯合 Rosenblatt 轉換：z₁ = u₁，z₂ = C_{2|1}(u₂|u₁; IS copula)
z_rosen = np.column_stack([
    u_oos_pit[:, 0],
    bc_is.hfunc1(u_oos_pit).ravel()
])
z_rosen = np.clip(z_rosen, 1e-10, 1 - 1e-10)


def _rj_tag(p):
    return '  （拒絕）' if p < 0.05 else '  （未拒絕）'


def _print_pit_block(u_mat, var_labels, title):
    """列印一組 Rosenblatt PIT 的 LB 檢定結果。"""
    print(f'\n  ── {title} ──')
    for ai, lbl in enumerate(var_labels):
        col = u_mat[:, ai]
        v   = stats.norm.ppf(col)
        lbm = float(acorr_ljungbox(v,    lags=[20], return_df=True)['lb_pvalue'].iloc[-1])
        lbv = float(acorr_ljungbox(v**2, lags=[20], return_df=True)['lb_pvalue'].iloc[-1])
        print(f'\n    {lbl}：')
        print(f'      LB(20)  Φ⁻¹(u) 串列:     p={lbm:.4f}{_rj_tag(lbm)}')
        print(f'      LB²(20) [Φ⁻¹(u)]² ARCH:  p={lbv:.4f}{_rj_tag(lbv)}')


print(f'\n── Rosenblatt OOS PIT 檢定（無 look-ahead，IS 擬合 {copula_list[best_ci_out]} copula）──')
print('  H₀：序列為 i.i.d. U(0,1)；p < 0.05 拒絕 H₀')

_print_pit_block(z_rosen,   ['z₁ = u₁（邊際）', 'z₂ = C₂|₁(u₂|u₁)（條件）'],
                 '聯合 Rosenblatt U(0,1) 檢定（OOS，文獻標準）')


# ─────────────────────────────────────────────────────────────────────────────
# ── (2) Copula-only OOS DM：各 Copula 增益 vs 獨立基準 ───────────────────────
# ─────────────────────────────────────────────────────────────────────────────

print('\n── Copula-only OOS DM（log c(u₁,u₂) vs 獨立基準）──')
print('  各 copula 以 IS 邊際 PIT 固定擬合後評估 OOS log c（無前視）')
print('  損失差 dₜ = −log c(u₁ₜ,u₂ₜ)；H₁(gain): E[d]<0 ⟺ copula 優於獨立（* p<0.05）')
print(f'\n  {"Copula":12s}  {"avg log c":>12s}  {"DM stat":>10s}  {"p(two)":>8s}  {"p(gain)":>8s}')
print('  ' + '-' * 58)

u_oos_c = np.clip(u_oos_pit, 1e-6, 1 - 1e-6)
for ci, cop_name in enumerate(copula_list):
    try:
        bc_c = pv.Bicop(family=_FAMILY_MAP_PD[cop_name])
        bc_c.fit(np.clip(u, 1e-6, 1 - 1e-6))
        lc   = np.clip(np.log(np.maximum(bc_c.pdf(u_oos_c), 1e-300)), -50.0, 50.0)
    except Exception:
        lc   = np.zeros(T_pred)
    avg_lc        = float(np.mean(lc))
    dm_s, p2, pl  = _dm_test(-lc, np.zeros(T_pred), h_horizon=1)
    note = ' ← 最佳' if ci == best_ci_out else ''
    print(f'  {cop_name:12s}  {avg_lc:12.6f}  {dm_s:10.4f}  {p2:8.4f}  {pl:8.4f}{_sig(pl)}{note}')

print('\n  （avg log c > 0：OOS 聯合密度優於獨立；p(gain) < 0.05：copula 增益統計顯著）')








# ── FKO 共用參數與輔助函式 ──────────────────────────────────────────────────
from scipy.optimize import brentq as _brentq

RF_DAILY = 0.0
TC_BPS   = 10.0 / 1e4
GAMMAS   = [1.0, 5.0, 10.0]
ANN      = 252

def _build_cov(rho_series, h_mat):
    """回傳 (T_pred, 2, 2)；對角=h_mat，非對角=rho*sqrt(h1*h2)。"""
    T_  = len(rho_series)
    cov = np.zeros((T_, 2, 2))
    cov[:, 0, 0] = h_mat[:, 0]
    cov[:, 1, 1] = h_mat[:, 1]
    cov[:, 0, 1] = rho_series * np.sqrt(h_mat[:, 0] * h_mat[:, 1])
    cov[:, 1, 0] = cov[:, 0, 1]
    return cov

_zis_res     = np.column_stack([y[1:, 0] - drift_is[:, 0],
                                 y[1:, 1] - drift_is[:, 1]])
cov_const_is = np.cov(_zis_res.T)

def _portfolio(cov_seq, mu_t, mu_tgt):
    """逐期解 w_risky = mu_tgt * Σ⁻¹μ_ex / (μ_ex'Σ⁻¹μ_ex)；回傳 Rp, turnover, w_seq。"""
    ridge  = 1e-12 * np.eye(2)
    Rp_arr = np.zeros(T_pred)
    to_arr = np.zeros(T_pred)
    w_seq  = np.zeros((T_pred, 2))
    w_prev = np.zeros(2)
    for t in range(T_pred):
        mu_ex = mu_t[t] - RF_DAILY
        Sig   = cov_seq[t] + ridge
        try:
            Sinv  = np.linalg.inv(Sig)
            denom = float(mu_ex @ Sinv @ mu_ex)
            w_r   = mu_tgt * (Sinv @ mu_ex) / denom if denom > 1e-20 else np.zeros(2)
        except np.linalg.LinAlgError:
            w_r = np.zeros(2)
        w_seq[t]  = w_r
        to_arr[t] = float(np.sum(np.abs(w_r - w_prev)))
        w_prev    = w_r.copy()
        Rp_arr[t] = float(w_r @ y_pred[t]) + (1.0 - float(np.sum(w_r))) * RF_DAILY
    return Rp_arr, to_arr, w_seq

def _utility(Rp, gamma):
    g = gamma / (1.0 + gamma)
    return (1.0 + Rp) - 0.5 * g * (1.0 + Rp)**2

def _perf_fee(Rp_model, Rp_bench, gamma):
    f = lambda d: float(np.mean(_utility(Rp_model - d, gamma) - _utility(Rp_bench, gamma)))
    lo, hi = -0.01, 0.01
    fa, fb = f(lo), f(hi)
    for _ in range(12):
        if fa * fb < 0:
            break
        lo *= 2.0; hi *= 2.0
        fa, fb = f(lo), f(hi)
    if fa * fb < 0:
        try:
            return float(_brentq(f, lo, hi, xtol=1e-8))
        except Exception:
            pass
    dU   = float(np.mean(_utility(Rp_model, gamma) - _utility(Rp_bench, gamma)))
    dU_d = float(np.mean(-(1.0 - gamma/(1.0+gamma)*(1.0+Rp_model))))
    return (-dU / dU_d) if abs(dU_d) > 1e-20 else np.nan

def _ff(v, fmt='.4f'):
    return format(v, fmt) if not np.isnan(v) else 'nan'

def _ann_stats(Rp):
    mu_a  = float(np.mean(Rp)) * ANN
    vol_a = float(np.std(Rp, ddof=1)) * np.sqrt(ANN)
    sr    = mu_a / vol_a if vol_a > 1e-12 else np.nan
    return mu_a, vol_a, sr

mu_target = float(np.mean(drift_is))

# ── 兩個獨立單市場 SV IS（C 對照基準：獨立重取樣，無 copula）────────────────
print(f'正在跑獨立單市場 IS 粒子濾波（N={N}, T={T}）…', flush=True)

# ── Asset 1 ──────────────────────────────────────────────────────────────
np.random.seed(43)
_r2_s1 = np.random.randn(T, N)
_x1s   = np.zeros((T, N));    _th1s = np.zeros((T, N, k))
_m1s   = np.zeros((T, N, k)); _mu1s = np.zeros((T, N))
_w1s   = np.zeros((T, N))

_x1s[0] = np.random.normal(v1j, v1j*d, N)
_neg = _x1s[0] < 0;  _x1s[0, _neg] = 2*v1j - _x1s[0, _neg]
_mu1s[0] = _x1s[0].copy();  _w1s[0] = 1.0/N
_th1s[0,:,0] = np.random.normal(c1[0], v1_arr[0], N)
_th1s[0,:,1] = np.clip(np.random.gamma(c1[1]**2/v1_arr[1]**2, v1_arr[1]**2/c1[1], N), 1e-6, np.inf)
_th1s[0,:,2] = np.clip(np.random.gamma(v1j**2/v1_arr[2]**2,   v1_arr[2]**2/v1j,   N), 1e-6, np.inf)
_th1s[0,:,3] = np.clip(np.random.gamma(c1[3]**2/v1_arr[3]**2, v1_arr[3]**2/c1[3], N), 1e-6, np.inf)
_as1, _bs1 = c1[4]-np.sqrt(3)*v1_arr[4], c1[4]+np.sqrt(3)*v1_arr[4]
_th1s[0,:,4] = np.clip(np.random.uniform(_as1, _bs1, N), -0.999, 0.999)

for _t in range(1, T):
    _wp1 = _w1s[_t-1]
    # (1.1) 重抽樣前平滑
    _m1s[_t-1] = a*_th1s[_t-1] + (1-a)*np.sum(_wp1[:,None]*_th1s[_t-1], axis=0)
    # (1.2) 預測 mu
    _mu1s[_t] = _x1s[_t-1] + _m1s[_t-1,:,1]*(_m1s[_t-1,:,2]-_x1s[_t-1])
    # (1.3) g（單變量邊際密度）
    _MUg1  = _m1s[_t-1,:,0] - _mu1s[_t]/2.0
    _SIGg1 = np.sqrt(np.maximum(_mu1s[_t], 1e-12))
    _g1    = _wp1 * stats.norm.pdf(y[_t,0], _MUg1, _SIGg1)
    _g1s   = _g1.sum()
    _g1    = _g1/_g1s if _g1s > 0 else np.full(N, 1.0/N)
    # (2.1) 獨立重取樣（只動 asset 1 粒子）
    _rs1 = np.random.choice(N, N, replace=True, p=_g1)
    _x1s[_t-1]  = _x1s[_t-1, _rs1];   _th1s[_t-1] = _th1s[_t-1, _rs1]
    # (2.2) 重取樣後平滑
    _m1s[_t-1] = a*_th1s[_t-1] + (1-a)*np.mean(_th1s[_t-1], axis=0)
    _v1s = np.maximum(np.var(_th1s[_t-1], axis=0, ddof=1), v_all1)
    _mu1s[_t] = np.abs(_x1s[_t-1] + _m1s[_t-1,:,1]*(_m1s[_t-1,:,2]-_x1s[_t-1]))
    # (2.3) 抽樣新 theta1
    _th1s[_t,:,0] = np.random.normal(_m1s[_t-1,:,0], np.sqrt(h**2*_v1s[0]), N)
    for _pi in [1, 2, 3]:
        _ss = np.sqrt(h**2*_v1s[_pi])
        _th1s[_t,:,_pi] = np.clip(np.random.gamma(_m1s[_t-1,:,_pi]**2/_ss**2, _ss**2/_m1s[_t-1,:,_pi]), 1e-6, np.inf)
    _ss = np.sqrt(h**2*_v1s[4])
    _th1s[_t,:,4] = np.clip(np.random.uniform(_m1s[_t-1,:,4]-np.sqrt(3)*_ss, _m1s[_t-1,:,4]+np.sqrt(3)*_ss), -0.999, 0.999)
    # (2.4) SV 狀態更新
    _xp0 = np.maximum(_x1s[_t-1], 1e-12)
    _r1_ = (y[_t,0] - (_th1s[_t,:,0] - 0.5*_xp0)) / np.sqrt(_xp0)
    _x1s[_t] = (_xp0 + _th1s[_t,:,1]*(_th1s[_t,:,2]-_xp0)
                + _th1s[_t,:,3]*np.sqrt(_xp0)*(_th1s[_t,:,4]*_r1_
                  + np.sqrt(np.maximum(1-_th1s[_t,:,4]**2, 0))*_r2_s1[_t]))
    _x1s[_t] = np.abs(_x1s[_t])
    # (3.1) 重要性權重（單變量）
    _MUn1  = _th1s[_t,:,0] - _x1s[_t]/2.0
    _SIGn1 = np.sqrt(np.maximum(_x1s[_t], 1e-12))
    _MUp1  = _m1s[_t-1,:,0] - _mu1s[_t]/2.0
    _SIGp1 = np.sqrt(np.maximum(_mu1s[_t], 1e-12))
    _ln1   = stats.norm.pdf(y[_t,0], _MUn1, _SIGn1)
    _lp1   = stats.norm.pdf(y[_t,0], _MUp1, _SIGp1)
    _w1s[_t] = _ln1 / np.maximum(_lp1, 1e-300)
    _wsum = _w1s[_t].sum()
    _w1s[_t] = _w1s[_t]/_wsum if _wsum > 0 else np.full(N, 1.0/N)

# ── Asset 2 ──────────────────────────────────────────────────────────────
np.random.seed(44)
_r2_s2 = np.random.randn(T, N)
_x2s   = np.zeros((T, N));    _th2s = np.zeros((T, N, k))
_m2s   = np.zeros((T, N, k)); _mu2s = np.zeros((T, N))
_w2s   = np.zeros((T, N))

_x2s[0] = np.random.normal(v2j, v2j*d, N)
_neg = _x2s[0] < 0;  _x2s[0, _neg] = 2*v2j - _x2s[0, _neg]
_mu2s[0] = _x2s[0].copy();  _w2s[0] = 1.0/N
_th2s[0,:,0] = np.random.normal(c2[0], v2_arr[0], N)
_th2s[0,:,1] = np.clip(np.random.gamma(c2[1]**2/v2_arr[1]**2, v2_arr[1]**2/c2[1], N), 1e-6, np.inf)
_th2s[0,:,2] = np.clip(np.random.gamma(v2j**2/v2_arr[2]**2,   v2_arr[2]**2/v2j,   N), 1e-6, np.inf)
_th2s[0,:,3] = np.clip(np.random.gamma(c2[3]**2/v2_arr[3]**2, v2_arr[3]**2/c2[3], N), 1e-6, np.inf)
_as2, _bs2 = c2[4]-np.sqrt(3)*v2_arr[4], c2[4]+np.sqrt(3)*v2_arr[4]
_th2s[0,:,4] = np.clip(np.random.uniform(_as2, _bs2, N), -0.999, 0.999)

for _t in range(1, T):
    _wp2 = _w2s[_t-1]
    # (1.1) 重抽樣前平滑
    _m2s[_t-1] = a*_th2s[_t-1] + (1-a)*np.sum(_wp2[:,None]*_th2s[_t-1], axis=0)
    # (1.2) 預測 mu
    _mu2s[_t] = _x2s[_t-1] + _m2s[_t-1,:,1]*(_m2s[_t-1,:,2]-_x2s[_t-1])
    # (1.3) g（單變量邊際密度）
    _MUg2  = _m2s[_t-1,:,0] - _mu2s[_t]/2.0
    _SIGg2 = np.sqrt(np.maximum(_mu2s[_t], 1e-12))
    _g2    = _wp2 * stats.norm.pdf(y[_t,1], _MUg2, _SIGg2)
    _g2s   = _g2.sum()
    _g2    = _g2/_g2s if _g2s > 0 else np.full(N, 1.0/N)
    # (2.1) 獨立重取樣（只動 asset 2 粒子）
    _rs2 = np.random.choice(N, N, replace=True, p=_g2)
    _x2s[_t-1]  = _x2s[_t-1, _rs2];   _th2s[_t-1] = _th2s[_t-1, _rs2]
    # (2.2) 重取樣後平滑
    _m2s[_t-1] = a*_th2s[_t-1] + (1-a)*np.mean(_th2s[_t-1], axis=0)
    _v2s = np.maximum(np.var(_th2s[_t-1], axis=0, ddof=1), v_all2)
    _mu2s[_t] = np.abs(_x2s[_t-1] + _m2s[_t-1,:,1]*(_m2s[_t-1,:,2]-_x2s[_t-1]))
    # (2.3) 抽樣新 theta2
    _th2s[_t,:,0] = np.random.normal(_m2s[_t-1,:,0], np.sqrt(h**2*_v2s[0]), N)
    for _pi in [1, 2, 3]:
        _ss = np.sqrt(h**2*_v2s[_pi])
        _th2s[_t,:,_pi] = np.clip(np.random.gamma(_m2s[_t-1,:,_pi]**2/_ss**2, _ss**2/_m2s[_t-1,:,_pi]), 1e-6, np.inf)
    _ss = np.sqrt(h**2*_v2s[4])
    _th2s[_t,:,4] = np.clip(np.random.uniform(_m2s[_t-1,:,4]-np.sqrt(3)*_ss, _m2s[_t-1,:,4]+np.sqrt(3)*_ss), -0.999, 0.999)
    # (2.4) SV 狀態更新
    _xp1 = np.maximum(_x2s[_t-1], 1e-12)
    _r1_ = (y[_t,1] - (_th2s[_t,:,0] - 0.5*_xp1)) / np.sqrt(_xp1)
    _x2s[_t] = (_xp1 + _th2s[_t,:,1]*(_th2s[_t,:,2]-_xp1)
                + _th2s[_t,:,3]*np.sqrt(_xp1)*(_th2s[_t,:,4]*_r1_
                  + np.sqrt(np.maximum(1-_th2s[_t,:,4]**2, 0))*_r2_s2[_t]))
    _x2s[_t] = np.abs(_x2s[_t])
    # (3.1) 重要性權重（單變量）
    _MUn2  = _th2s[_t,:,0] - _x2s[_t]/2.0
    _SIGn2 = np.sqrt(np.maximum(_x2s[_t], 1e-12))
    _MUp2  = _m2s[_t-1,:,0] - _mu2s[_t]/2.0
    _SIGp2 = np.sqrt(np.maximum(_mu2s[_t], 1e-12))
    _ln2   = stats.norm.pdf(y[_t,1], _MUn2, _SIGn2)
    _lp2   = stats.norm.pdf(y[_t,1], _MUp2, _SIGp2)
    _w2s[_t] = _ln2 / np.maximum(_lp2, 1e-300)
    _wsum = _w2s[_t].sum()
    _w2s[_t] = _w2s[_t]/_wsum if _wsum > 0 else np.full(N, 1.0/N)

# IS T-1 後驗均值（兩市場獨立）
_th1f_s = np.mean(_th1s[T-1], axis=0)          # shape (k,)
_th2f_s = np.mean(_th2s[T-1], axis=0)          # shape (k,)
_xf_s   = np.array([np.mean(_x1s[T-1]), np.mean(_x2s[T-1])])  # shape (2,)
print('獨立單市場 IS 完成')


# ═══════════════════════════════════════════════════════════════════════════
# ──  FKO：每個模型用自己的完整參數集（μ AND Σ）─────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _run_oos_filter(th1_f, th2_f, x_f):
    """OOS 確定性均值演進：輸入為 1D 點估計向量，只保留資料驅動項，無隨機噪聲。
    th1_f, th2_f: shape (k,)；x_f: shape (2,)
    回傳 (mu1_oos, mu2_oos, h_roll (T_pred,2), drift_oos (T_pred,2))。"""
    _mu1, _k1, _tl1, _s1, _rh1 = (float(th1_f[0]), float(th1_f[1]), float(th1_f[2]),
                                    float(th1_f[3]), float(th1_f[4]))
    _mu2, _k2, _tl2, _s2, _rh2 = (float(th2_f[0]), float(th2_f[1]), float(th2_f[2]),
                                    float(th2_f[3]), float(th2_f[4]))
    _x   = np.array([float(x_f[0]), float(x_f[1])])

    _xst = np.empty((T_pred, 2))
    _hr  = np.empty((T_pred, 2))
    for _t in range(T_pred):
        _xst[_t] = _x
        _hr[_t]  = np.maximum(_x, 1e-10)
        if _t < T_pred - 1:
            _sx1 = float(np.sqrt(max(_x[0], 1e-8)))
            _sx2 = float(np.sqrt(max(_x[1], 1e-8)))
            _ra  = (y_pred[_t,0] - (_mu1 - 0.5*_x[0])) / _sx1
            _rb  = (y_pred[_t,1] - (_mu2 - 0.5*_x[1])) / _sx2
            _x[0] = max(_x[0] + _k1*(_tl1 - _x[0]) + _s1*_sx1*_rh1*_ra, 1e-10)
            _x[1] = max(_x[1] + _k2*(_tl2 - _x[1]) + _s2*_sx2*_rh2*_rb, 1e-10)

    _drift = np.column_stack([
        _mu1 - 0.5*_xst[:,0],
        _mu2 - 0.5*_xst[:,1]
    ])
    return _mu1, _mu2, _hr, _drift


def _fko_perf_fee(Rp_model, Rp_bench, gamma):
    return _perf_fee(Rp_model, Rp_bench, gamma)


print('\n' + '═'*78)
print(' FKO 投資組合經濟價值評估 ')
print('═'*78)
print('  【設計說明】')
print('  模型              IS 後驗來源                         Σ')
print('  SV-Copula       → 聯合 IS（best copula）T-1 後驗      ρ_const(best copula) + 各自 h_roll')
print('  Separate SV     → 兩個獨立單市場 IS T-1 後驗           ρ=0 + 各自 h_roll（獨立重取樣）')
print('  SepSV+ρ_cop     → 兩個獨立單市場 IS T-1 後驗           ρ_const(best copula) + SepSV h_roll')
print('  OOS h_roll：IS 後驗折疊為均值點估計，確定性演進（只保留資料驅動項，無噪聲）')
print('  SV-Copula vs Separate SV：涵蓋 ① copula ρ  ② 聯合 vs 獨立重取樣 兩效果')
print('  SepSV+ρ_cop  vs Separate SV：單獨分離 ① copula ρ 的貢獻（參數來源相同）')

_th1_c2 = theta1_s[T-1, :, best_ci_out]   # shape (k,)，全序列粒子均值
_th2_c2 = theta2_s[T-1, :, best_ci_out]   # shape (k,)
_xf_c2  = x_s[T-1, :, best_ci_out]        # shape (p,)

_mu1_c2, _mu2_c2, _hr_c2, _dr_c2 = _run_oos_filter(_th1_c2, _th2_c2, _xf_c2)
_mu1_s2, _mu2_s2, _hr_s2, _dr_s2 = _run_oos_filter(_th1f_s, _th2f_s, _xf_s)

print(f'\n  μ_oos 資產1: SV-Copula={_mu1_c2:.6f}  Separate SV={_mu1_s2:.6f}  差={_mu1_c2-_mu1_s2:+.6f}')
print(f'  μ_oos 資產2: SV-Copula={_mu2_c2:.6f}  Separate SV={_mu2_s2:.6f}  差={_mu2_c2-_mu2_s2:+.6f}')
_tau2f  = float(R_mat[0, best_ci_out])
_rho2f  = float(np.clip(np.sin(np.pi * _tau2f / 2.0), -0.999, 0.999))
print(f"  ρ_const(SV-Copula) = {_rho2f:.6f}  ← {copula_list[best_ci_out]}  Kendall τ = {_tau2f:.6f}")

_mu_t_c2 = np.tile(np.mean(_dr_c2, axis=0), (T_pred, 1))
_mu_t_s2 = np.tile(np.mean(_dr_s2, axis=0), (T_pred, 1))

_cov_c2 = _build_cov(np.full(T_pred, _rho2f), _hr_c2)
_cov_s2 = _build_cov(np.zeros(T_pred),         _hr_s2)
_cov_h2 = _build_cov(np.full(T_pred, _rho2f), _hr_s2)   # SepSV 參數 + copula ρ

_mu_tgt3 = float(np.mean(drift_is))

_Rp_c2, _To_c2, _ = _portfolio(_cov_c2, _mu_t_c2, _mu_tgt3)
_Rp_s2, _To_s2, _ = _portfolio(_cov_s2, _mu_t_s2, _mu_tgt3)
_Rp_h2, _To_h2, _ = _portfolio(_cov_h2, _mu_t_s2, _mu_tgt3)   # μ 同 SepSV，只換 ρ

_pc3 = {
    'SV-Copula':   (_Rp_c2, _To_c2),
    'Separate SV': (_Rp_s2, _To_s2),
    'SepSV+ρ_cop': (_Rp_h2, _To_h2),
}

print()
_MW3 = [14, 7, 12, 12, 10]
_MH3 = ['模型', '含TC', '年化均值%', '年化波動%', 'Sharpe']
print('── 表一：各模型投資組合年化績效（各行為該模型自身 Rp，各自 μ 與 Σ）──')
print('  ' + '  '.join(_rj(h, w) for h, w in zip(_MH3, _MW3)))
print('  ' + '  '.join('-'*w for w in _MW3))
for _mn3, (_Rr3, _Tr3) in _pc3.items():
    for _wtc3, _tl3 in [(False, '否'), (True, '是')]:
        _Re3 = _Rr3 - TC_BPS * _Tr3 if _wtc3 else _Rr3.copy()
        _a3, _v3, _s3 = _ann_stats(_Re3)
        print('  ' + '  '.join([
            _lj(_mn3,        _MW3[0]),
            _rj(_tl3,        _MW3[1]),
            _rj(_ff(_a3*100),_MW3[2]),
            _rj(_ff(_v3*100),_MW3[3]),
            _rj(_ff(_s3),    _MW3[4]),
        ]))
    print()

_FW3 = [4, 14, 7, 11, 11, 9]
_FH3 = ['γ', '模型', '含TC', 'Δ(bps)', 'DM-stat', 'p(gain)']
print('── 表二：FKO 年化轉換費 Δ（各模型 vs Separate SV 基準）──')
print('  Δ > 0：投資人願由 Separate SV 切換至該模型')
print('  p(gain)：效用差 DM 單側 p 值（* p<0.05 → 模型效用顯著較高）')
print()
print('  ' + '  '.join(_rj(h, w) for h, w in zip(_FH3, _FW3)))
print('  ' + '  '.join('-'*w for w in _FW3))

_Rb3r_s, _Tb3r_s = _pc3['Separate SV']
for _gm3 in GAMMAS:
    for _mn3 in ['SV-Copula', 'SepSV+ρ_cop']:
        _Rm3r, _Tm3r = _pc3[_mn3]
        for _tc3, _tcl3 in [(False, '否'), (True, '是')]:
            _Rm3 = _Rm3r - TC_BPS * _Tm3r if _tc3 else _Rm3r.copy()
            _Rb3 = _Rb3r_s - TC_BPS * _Tb3r_s if _tc3 else _Rb3r_s.copy()
            _dd3   = _fko_perf_fee(_Rm3, _Rb3, _gm3)
            _dbps3 = _dd3 * ANN * 1e4 if not np.isnan(_dd3) else np.nan
            _Um3   = _utility(_Rm3, _gm3)
            _Ub3   = _utility(_Rb3, _gm3)
            _ds3, _, _pg3_less = _dm_test(-_Ub3, -_Um3)
            _pg3 = 1.0 - _pg3_less
            print('  ' + '  '.join([
                _rj(f'{_gm3:.0f}',        _FW3[0]),
                _lj(_mn3,                 _FW3[1]),
                _rj(_tcl3,                _FW3[2]),
                _rj(_ff(_dbps3, '.2f'),   _FW3[3]),
                _rj(_ff(_ds3),            _FW3[4]),
                _rj(_ff(_pg3) + (_sig(_pg3) if not np.isnan(_pg3) else ''), _FW3[5]),
            ]))
    print()

print()
print('  【注腳】')
print('  (1) 靜態相依：copula 為 IS 固定擬合，ρ 為常數；非動態條件相依。')
if copula_list[best_ci_out] not in ('copulaN', 'copulaT'):
    print(f'  (2) Archimedean 近似：{copula_list[best_ci_out]} 使用 ρ=sin(πτ/2) 為近似轉換，詮釋需謹慎。')
print('  (3) Δ > 0 且 p(gain) < 0.05 → 聯合 SV-Copula 在本樣本有統計顯著正 EV。')
print('  (4) Δ 同時反映 copula 相依與聯合估計效果，為完整聯合框架 vs 獨立單市場之總經濟價值。')

# ═══════════════════════════════════════════════════════════════════════════════
# ── Mincer-Zarnowitz 型相依動態追蹤檢定（修正版）────────────────────────────────
# 設計：τ_model[t]（t 時點資訊）→ 預測 τ_realized[t]（未來 h 期已實現相依）
# IS/OOS τ_model 均用「擴張視窗重擬合同族 copula」，方法一致、可跨段比較
# τ_realized = 向前 MZ_WIN 期 Kendall τ（原始報酬，model-free，h=MZ_WIN）
# 標準誤：Newey-West HAC（lag=MZ_WIN-1），處理滾動視窗造成的強自相關
# IS 另附 SDPF rho_Ns 作為對照（估計子差異歸因）
# ═══════════════════════════════════════════════════════════════════════════════
MZ_WIN = 60
MZ_LAG = MZ_WIN - 1   # NW HAC lag = 視窗重疊期數

# ── IS PITs（best copula 平滑後驗重算，用於滾動重擬合）───────────────────────
_MU_mz  = (np.column_stack([theta1_s[:, 0, best_ci_out],
                              theta2_s[:, 0, best_ci_out]])
           - x_s[:, :, best_ci_out] / 2.0)
_SIG_mz = np.sqrt(np.maximum(x_s[:, :, best_ci_out], 1e-12))
_u_is_mz = np.clip(stats.norm.cdf(y, _MU_mz, _SIG_mz), 1e-6, 1 - 1e-6)

# ── IS τ_model A：滾動固定視窗重擬合（與 τ_realized 同尺度 MZ_WIN 期）──────────
_tau_mod_is_r = np.full(T, np.nan)
for _t in range(MZ_WIN, T):
    try:
        _bc_mz = pv.Bicop(family=_FAMILY_MAP_PD[copula_list[best_ci_out]])
        _bc_mz.fit(_u_is_mz[_t-MZ_WIN:_t])   # 固定回望 MZ_WIN 期
        _tau_mod_is_r[_t] = float(_bc_mz.tau)
    except Exception:
        pass

# ── IS τ_realized：向前 MZ_WIN 期 Kendall τ（原始報酬）─────────────────────
_tau_real_is = np.full(T, np.nan)
for _t in range(0, T - MZ_WIN):
    _kt, _ = stats.kendalltau(y[_t:_t+MZ_WIN, 0], y[_t:_t+MZ_WIN, 1])
    _tau_real_is[_t] = _kt

# ── OOS τ_model：滾動固定視窗重擬合 OOS PIT（固定回望 MZ_WIN 期）──────────────
_tau_mod_oos = np.full(T_pred, np.nan)
for _t in range(MZ_WIN, T_pred):
    try:
        _bc_mz = pv.Bicop(family=_FAMILY_MAP_PD[copula_list[best_ci_out]])
        _bc_mz.fit(u_oos_c[_t-MZ_WIN:_t])    # 固定回望 MZ_WIN 期
        _tau_mod_oos[_t] = float(_bc_mz.tau)
    except Exception:
        pass

# ── OOS τ_realized：向前 MZ_WIN 期 Kendall τ（原始報酬）────────────────────
_tau_real_oos = np.full(T_pred, np.nan)
for _t in range(0, T_pred - MZ_WIN):
    _kt, _ = stats.kendalltau(y_pred[_t:_t+MZ_WIN, 0], y_pred[_t:_t+MZ_WIN, 1])
    _tau_real_oos[_t] = _kt

# ── Newey-West HAC 共變異數矩陣 ─────────────────────────────────────────────
def _nw_cov(X, resid, lag):
    n   = len(resid)
    Xe  = X * resid[:, np.newaxis]          # score: x_t * e_t，shape (n, k)
    S   = Xe.T @ Xe / n
    for l in range(1, lag + 1):
        w     = 1.0 - l / (lag + 1)         # Bartlett kernel
        Gam   = Xe[l:].T @ Xe[:-l] / n
        S    += w * (Gam + Gam.T)
    XtXi = np.linalg.inv(X.T @ X / n)
    return XtXi @ S @ XtXi / n              # Var(β̂)

# ── OLS + HAC 迴歸 ───────────────────────────────────────────────────────────
def _mz_reg(x_arr, y_arr, lag):
    _mk = ~(np.isnan(x_arr) | np.isnan(y_arr))
    _x, _y = x_arr[_mk], y_arr[_mk]
    _n = len(_x)
    if _n < lag + 5:
        return None
    _X  = np.column_stack([np.ones(_n), _x])
    _bt = np.linalg.lstsq(_X, _y, rcond=None)[0]
    _e  = _y - _X @ _bt
    _ss_r = float(np.sum(_e**2))
    _ss_t = float(np.sum((_y - _y.mean())**2))
    _r2   = 1.0 - _ss_r / _ss_t if _ss_t > 1e-20 else np.nan
    _V    = _nw_cov(_X, _e, lag)
    _se   = np.sqrt(np.maximum(np.diag(_V), 0))
    _tab  = _bt / np.maximum(_se, 1e-20)
    _pab  = [2.0 * float(stats.norm.sf(abs(_tv))) for _tv in _tab]   # HAC 漸近常態
    _df   = np.array([_bt[0], _bt[1] - 1.0])   # H₀: a=0, b=1
    try:
        _W  = float(_df @ np.linalg.inv(_V) @ _df)
        _pW = float(stats.chi2.sf(_W, 2))
    except np.linalg.LinAlgError:
        _W, _pW = np.nan, np.nan
    _z_b1 = (_bt[1] - 1.0) / max(_se[1], 1e-20)        # L2: H₀ b=1，a 自由
    _p_b1 = 2.0 * float(stats.norm.sf(abs(_z_b1)))
    _sign = float(np.mean(                               # L3/4: 符號一致率
        np.sign(_x - np.mean(_x)) == np.sign(_y - np.mean(_y))))
    return {'a': _bt[0], 'b': _bt[1], 'se_a': _se[0], 'se_b': _se[1],
            't_a': _tab[0], 't_b': _tab[1], 'p_a': _pab[0], 'p_b': _pab[1],
            'z_b1': _z_b1, 'p_b1': _p_b1, 'sign': _sign,
            'r2': _r2, 'W': _W, 'p_W': _pW, 'n': _n}

def _mz_print(res, label):
    if res is None:
        print(f'  {label}: 有效樣本不足')
        return
    print(f'  {label}  (n={res["n"]}，重疊版，HAC lag={MZ_LAG})')
    print(f'  [L3/4 有訊號?]')
    print(f'    b=0  z={res["t_b"]:6.2f}  p={res["p_b"]:.4f}  [H₀: b=0]'
          + ('  *' if res['p_b'] < 0.05 else ''))
    print(f'    符號一致率 = {res["sign"]:.4f}')
    print(f'  [L2 幅度正確?]')
    print(f'    b=1  z={res["z_b1"]:6.2f}  p={res["p_b1"]:.4f}  [H₀: b=1, a 自由]')
    print(f'    截距 a={res["a"]:7.4f}  HAC-se={res["se_a"]:.4f}  z={res["t_a"]:6.2f}  p={res["p_a"]:.4f}')
    print(f'    斜率 b={res["b"]:7.4f}  HAC-se={res["se_b"]:.4f}')
    print(f'    Wald(a=0,b=1) χ²(2)={res["W"]:7.3f}  p={res["p_W"]:.4f}  R²={res["r2"]:.4f}')
    print()

# ── L5：HAC DM 檢定（d = e_naive²−e_model²；stat>0 → model wins）────────────
def _mz_dm(e_model, e_naive, lag):
    _mk = ~(np.isnan(e_model) | np.isnan(e_naive))
    _em, _en = e_model[_mk], e_naive[_mk]
    _n = len(_em)
    if _n < lag + 5:
        return np.nan, np.nan
    _d  = _en**2 - _em**2
    _dm = np.mean(_d)
    _dc = _d - _dm
    _s2 = np.sum(_dc**2) / _n
    for l in range(1, lag + 1):
        _s2 += 2 * (1 - l/(lag+1)) * np.sum(_dc[l:]*_dc[:-l]) / _n
    _se  = np.sqrt(max(_s2, 0) / _n)
    _stat = _dm / max(_se, 1e-20)
    return _stat, float(stats.norm.sf(_stat))   # 單尾：p(model wins)

# ── L5：HLN encompassing（enc = e_naive(e_naive−e_model)；stat>0 → model encp naive）
def _mz_enc(e_model, e_naive, lag):
    _mk = ~(np.isnan(e_model) | np.isnan(e_naive))
    _em, _en = e_model[_mk], e_naive[_mk]
    _n = len(_em)
    if _n < lag + 5:
        return np.nan, np.nan
    _enc = _en * (_en - _em)
    _em2 = np.mean(_enc)
    _ec  = _enc - _em2
    _s2  = np.sum(_ec**2) / _n
    for l in range(1, lag + 1):
        _s2 += 2 * (1 - l/(lag+1)) * np.sum(_ec[l:]*_ec[:-l]) / _n
    _se  = np.sqrt(max(_s2, 0) / _n)
    _stat = _em2 / max(_se, 1e-20)
    return _stat, float(stats.norm.sf(_stat))   # 單尾：p(model encompasses naive)

# ── L5：輸出 ──────────────────────────────────────────────────────────────────
def _mz_lv5(tau_mod, tau_real, tau_const, tau_rw, lag):
    _e_mod   = tau_real - tau_mod
    _e_const = tau_real - tau_const        # tau_const 為純量
    _e_rw    = tau_real - tau_rw
    print(f'  [L5 打敗天真?]')
    _dm_c, _p_c   = _mz_dm(_e_mod, _e_const, lag)
    _ec_c, _pe_c  = _mz_enc(_e_mod, _e_const, lag)
    _dm_r, _p_r   = _mz_dm(_e_mod, _e_rw, lag)
    _ec_r, _pe_r  = _mz_enc(_e_mod, _e_rw, lag)
    _s = lambda p: '  *' if (not np.isnan(p)) and p < 0.05 else ''
    print(f'    vs 常數基準  : DM z={_dm_c:6.2f} p={_p_c:.4f}{_s(_p_c)}'
          f'  ENC z={_ec_c:6.2f} p={_pe_c:.4f}{_s(_pe_c)}')
    print(f'    vs 已實現外推: DM z={_dm_r:6.2f} p={_p_r:.4f}{_s(_p_r)}'
          f'  ENC z={_ec_r:6.2f} p={_pe_r:.4f}{_s(_pe_r)}')
    print()

# ── 非重疊版（每 MZ_WIN 步取一點，消除重疊自相關，OLS + F 檢定）───────────────
def _mz_reg_noovlp(x_arr, y_arr):
    _mk  = ~(np.isnan(x_arr) | np.isnan(y_arr))
    _all = np.where(_mk)[0]
    if len(_all) == 0:
        return None
    _idx = _all[::MZ_WIN]   # 每 MZ_WIN 步取一點，確保非重疊
    _x, _y = x_arr[_idx], y_arr[_idx]
    _n = len(_x)
    if _n < 4:
        return None
    _X   = np.column_stack([np.ones(_n), _x])
    _bt  = np.linalg.lstsq(_X, _y, rcond=None)[0]
    _e   = _y - _X @ _bt
    _s2  = float(np.sum(_e**2)) / max(_n - 2, 1)
    _se  = np.sqrt(np.maximum(np.diag(_s2 * np.linalg.inv(_X.T @ _X)), 0))
    _tab = _bt / np.maximum(_se, 1e-20)
    _pab = [2.0 * float(stats.t.sf(abs(_tv), df=_n - 2)) for _tv in _tab]
    _ss_r = float(np.sum(_e**2))
    _ss_t = float(np.sum((_y - _y.mean())**2))
    _r2   = 1.0 - _ss_r / _ss_t if _ss_t > 1e-20 else np.nan
    _df   = np.array([_bt[0], _bt[1] - 1.0])
    _Fs   = float(_df @ (_X.T @ _X) @ _df) / (2.0 * _s2) if _s2 > 1e-20 else np.nan
    _pF   = float(stats.f.sf(_Fs, 2, _n - 2)) if not np.isnan(_Fs) else np.nan
    return {'a': _bt[0], 'b': _bt[1], 'se_a': _se[0], 'se_b': _se[1],
            't_a': _tab[0], 't_b': _tab[1], 'p_a': _pab[0], 'p_b': _pab[1],
            'r2': _r2, 'F': _Fs, 'p_F': _pF, 'n': _n}

def _mz_print_noovlp(res, label):
    if res is None:
        print(f'  {label}: 有效樣本不足（n<4）')
        return
    _warn = '  ⚠ 有效獨立樣本過少，p 值僅供參考' if res['n'] < 10 else ''
    print(f'  {label}  (n_eff={res["n"]}，非重疊，OLS+t/F){_warn}')
    print(f'    截距 a = {res["a"]:8.4f}  se={res["se_a"]:.4f}  t={res["t_a"]:6.2f}  p={res["p_a"]:.4f}')
    print(f'    斜率 b = {res["b"]:8.4f}  se={res["se_b"]:.4f}  t={res["t_b"]:6.2f}  p={res["p_b"]:.4f}')
    print(f'    R²    = {res["r2"]:8.4f}')
    print(f'    F(a=0,b=1, 2,{res["n"]-2}) = {res["F"]:.3f}  p={res["p_F"]:.4f}')
    print()

# ── 天真基準（L5 用）─────────────────────────────────────────────────────────
# 常數基準：IS τ_realized 樣本內均值（OOS 段亦用此值，保持樣本外性質）
_tau_const_is  = float(np.nanmean(_tau_real_is))
_tau_const_oos = _tau_const_is

# 已實現外推（random walk）：過去 MZ_WIN 期原始報酬的 Kendall τ
_tau_rw_is  = np.full(T, np.nan)
for _t in range(MZ_WIN, T):
    _kt, _ = stats.kendalltau(y[_t-MZ_WIN:_t, 0], y[_t-MZ_WIN:_t, 1])
    _tau_rw_is[_t] = _kt

_tau_rw_oos = np.full(T_pred, np.nan)
for _t in range(MZ_WIN, T_pred):
    _kt, _ = stats.kendalltau(y_pred[_t-MZ_WIN:_t, 0], y_pred[_t-MZ_WIN:_t, 1])
    _tau_rw_oos[_t] = _kt

print('\n' + '═'*78)
print(' Mincer-Zarnowitz 型相依動態追蹤檢定（修正版）')
print('═'*78)
print(f'  τ_model    = 滾動 {MZ_WIN} 期固定視窗重擬合 {copula_list[best_ci_out]}（IS/OOS 同方法）')
print(f'  τ_realized = 向前 {MZ_WIN} 期 Kendall τ（原始報酬，model-free）')
print(f'  重疊版：NW HAC（lag={MZ_LAG}），z 統計量，Wald χ²(2)')
print(f'  非重疊版：每 {MZ_WIN} 步取一點（n_eff≈n/{MZ_WIN}），OLS t/F（真實有效自由度）')
print(f'  迴歸：τ_realized[t, t+h] = a + b·τ_model[t-h, t] + ε  （h={MZ_WIN}）')
print()

print(f'── IS 段：τ_model = 滾動重擬合 ──')
_mz_print(_mz_reg(_tau_mod_is_r, _tau_real_is, MZ_LAG),
          f'{copula_list[best_ci_out]} IS-refit（重疊）')
_mz_lv5(_tau_mod_is_r, _tau_real_is, _tau_const_is, _tau_rw_is, MZ_LAG)
_mz_print_noovlp(_mz_reg_noovlp(_tau_mod_is_r, _tau_real_is),
                 f'{copula_list[best_ci_out]} IS-refit')

print(f'── OOS 段：τ_model = 滾動重擬合 OOS PIT ──')
_mz_print(_mz_reg(_tau_mod_oos, _tau_real_oos, MZ_LAG),
          f'{copula_list[best_ci_out]} OOS-refit（重疊）')
_mz_lv5(_tau_mod_oos, _tau_real_oos, _tau_const_oos, _tau_rw_oos, MZ_LAG)
_mz_print_noovlp(_mz_reg_noovlp(_tau_mod_oos, _tau_real_oos),
                 f'{copula_list[best_ci_out]} OOS-refit')
