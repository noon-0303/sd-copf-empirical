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

N = 500
k = 5
p = 2
h = T * 1e-4
a = np.sqrt(1.0 - h**2)

copula_list = ["copulaN", "copulaT", "copulaC", "copulaF", "copulaG"]
n_cop = len(copula_list)

FORCE_COPULA = "copulaF"   # 暫時強制固定使用 copulaF；還原成 logLik 篩選設為 None

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
        u_ll   = np.clip(stats.norm.cdf(y, MU_ll, SIG_ll), 1e-6, 1 - 1e-6)
        yp_ll  = stats.norm.pdf(y, MU_ll, SIG_ll)
        cop_ll, rho_ll = copulafitall23(copula_list, c, u_ll)
        jpdf   = np.maximum(cop_ll * yp_ll[:, 0] * yp_ll[:, 1], np.finfo(float).tiny)
        logLik[j, ci]     = np.sum(np.log(jpdf))

        _is_selected = (c == FORCE_COPULA) if FORCE_COPULA is not None \
            else (x_best is None or logLik[j, ci] > logLik[j, best_ci_so_far])

        if _is_selected:
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

    best_ci = copula_list.index(FORCE_COPULA) if FORCE_COPULA is not None else int(np.argmax(logLik[j]))
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
plt.tight_layout()
fig.savefig('is_returns_and_volatility.png', dpi=150, bbox_inches='tight')
plt.show()


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
    pi   = (n01 + n11) / (len(v)-1)
    if pi in (0.0, 1.0) or pi01 in (0.0, 1.0) or pi11 in (0.0, 1.0):
        return np.nan, np.nan
    lr = -2.0 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
        - n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )
    return lr, float(1.0 - stats.chi2.cdf(lr, df=1))


# 表格
def _dw(s):
    """字串顯示寬度（CJK 全形字元算 2，其餘算 1）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in str(s))

def _lj(s, w): return str(s) + ' ' * max(0, w - _dw(str(s)))
def _rj(s, w): return ' ' * max(0, w - _dw(str(s))) + str(s)
def _fl(v):    return f'{v:.4f}' if not np.isnan(v) else 'nan'
def _sig(p):   return '*' if (not np.isnan(p) and p < 0.05) else ' '

WCOL = [  14,     9,     7,      8,     7,      8,     7]
HCOL = ['資產', '違反/T', '違反率',
        'LR_POF', 'p_POF', 'LR_ind', 'p_ind']







# ─────────────────────────────────────────────────────────────────────────────
# ────── Volatility Forecasting & QLIKE ───────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


PRED_PATH = r"C:\Users\user\PycharmProjects\sd-copf\Empirical\data\real_data2-1_pred.xlsx"
y_pred1 = pd.read_excel(PRED_PATH, sheet_name='return1', header=None).values.ravel()
y_pred2 = pd.read_excel(PRED_PATH, sheet_name='return2', header=None).values.ravel()
y_pred  = np.column_stack([y_pred1, y_pred2])
T_pred  = len(y_pred1)


# ─────────────────────────────────────────────────────────────────────────────
# ── 牛熊市判定（峰谷回撤法，20% drawdown rule）────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

BEAR_THRESHOLD = 0.20   # 峰值下跌 20% 判定進入熊市，谷底反彈 20% 判定進入牛市

def _classify_bull_bear(price, threshold=BEAR_THRESHOLD):
    """峰谷回撤法：回傳每期市場狀態（'bull'/'bear'）。
    熊市起點回溯至前一個波峰，牛市起點回溯至前一個波谷。"""
    n = len(price)
    turning_points = []          # (index, 'peak' | 'trough')
    mode = 'up'
    ext_i, ext_v = 0, price[0]
    for t in range(1, n):
        if mode == 'up':
            if price[t] > ext_v:
                ext_i, ext_v = t, price[t]
            elif price[t] <= ext_v * (1 - threshold):
                turning_points.append((ext_i, 'peak'))
                mode = 'down'
                ext_i, ext_v = t, price[t]
        else:
            if price[t] < ext_v:
                ext_i, ext_v = t, price[t]
            elif price[t] >= ext_v * (1 + threshold):
                turning_points.append((ext_i, 'trough'))
                mode = 'up'
                ext_i, ext_v = t, price[t]

    state = np.empty(n, dtype=object)
    if not turning_points:
        state[:] = 'bull' if mode == 'up' else 'bear'
        return state

    first_i, first_kind = turning_points[0]
    state[:first_i + 1] = 'bull' if first_kind == 'peak' else 'bear'

    for (i0, k0), (i1, k1) in zip(turning_points[:-1], turning_points[1:]):
        state[i0:i1 + 1] = 'bear' if k0 == 'peak' else 'bull'

    last_i, last_kind = turning_points[-1]
    state[last_i:] = 'bear' if last_kind == 'peak' else 'bull'
    return state


def _segments_by_state(state, target):
    """回傳 state 中連續等於 target 的 (start, end) 索引清單（含頭尾）。"""
    segs, start = [], None
    for t, s in enumerate(state):
        if s == target and start is None:
            start = t
        elif s != target and start is not None:
            segs.append((start, t - 1))
            start = None
    if start is not None:
        segs.append((start, len(state) - 1))
    return segs


y_full     = np.concatenate([y, y_pred], axis=0)       # IS + OOS 完整報酬序列
price_full = np.exp(np.cumsum(y_full, axis=0))         # 由對數報酬重建價格指數（起始基準=1）
t_full     = np.arange(len(y_full))
oos_split  = T                                          # IS/OOS 分界索引（IS 共 T 期）

fig_regime, axes_regime = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for ai, aname in enumerate(['資產1', '資產2']):
    price_i = price_full[:, ai]
    state_i = _classify_bull_bear(price_i, BEAR_THRESHOLD)

    ax = axes_regime[ai]
    ax.plot(t_full, price_i, color='black', linewidth=1.0, label='價格指數（重建，起始=1）')
    for s, e in _segments_by_state(state_i, 'bull'):
        ax.axvspan(s, e, color='tab:green', alpha=0.15)
    for s, e in _segments_by_state(state_i, 'bear'):
        ax.axvspan(s, e, color='tab:red', alpha=0.15)
    ax.axvline(oos_split, color='blue', linestyle='--', linewidth=1.0, label='IS/OOS 分界')
    ax.set_title(f'{aname}：牛熊市判定（峰谷回撤法，門檻={BEAR_THRESHOLD:.0%}）')
    ax.set_ylabel('價格指數')
    ax.legend(loc='upper left', fontsize=8)

axes_regime[-1].set_xlabel('t（IS 全期 + OOS，虛線右側為 OOS）')
plt.tight_layout()
fig_regime.savefig('bull_bear_regime.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\n── 牛熊市判定摘要（峰谷回撤法，門檻={BEAR_THRESHOLD:.0%}）──')
for ai, aname in enumerate(['資產1', '資產2']):
    state_i    = _classify_bull_bear(price_full[:, ai], BEAR_THRESHOLD)
    bear_segs  = _segments_by_state(state_i, 'bear')
    bull_segs  = _segments_by_state(state_i, 'bull')
    n_bear     = sum(e - s + 1 for s, e in bear_segs)
    n_bull     = sum(e - s + 1 for s, e in bull_segs)
    bear_str   = ', '.join(f'[{s},{e}]' for s, e in bear_segs) if bear_segs else '無'
    print(f'  {aname}：熊市 {n_bear} 期（{n_bear/len(state_i):.1%}）、牛市 {n_bull} 期（{n_bull/len(state_i):.1%}）')
    print(f'    熊市區段：{bear_str}')


RV_WIN        = 10   # 一般窗長
RV_WIN_TRUE   = 20   # QLIKE 真值（rv_concat 置中窗，IS+OOS）窗長





# ────────────── DM 檢定通用函式 ────────────────────────────────────────────────────────

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






# ────────────── IS 1-step-ahead forecast（用於 IS QLIKE 對照）─────────────────────────────

# 逐期參數
kappa1_hat_t = theta1_best[:-1, 1]   # (T-1,)，不截斷，跟 h_roll 側一致
long1_hat_t  = theta1_best[:-1, 2]   # (T-1,)
kappa2_hat_t = theta2_best[:-1, 1]
long2_hat_t  = theta2_best[:-1, 2]


# 一步預測
h_is    = np.empty((T, 2))
h_is[0] = x_best[0]
h_is[1:, 0] = x_best[:-1, 0] + kappa1_hat_t * (long1_hat_t - x_best[:-1, 0])
h_is[1:, 1] = x_best[:-1, 1] + kappa2_hat_t * (long2_hat_t - x_best[:-1, 1])
h_is = np.maximum(h_is, 1e-10)


# 計算 IS部份 PIT
mu1_is_t = theta1_best[:-1, 0]
mu2_is_t = theta2_best[:-1, 0]
drift_is = np.column_stack([
    mu1_is_t - 0.5 * x_best[:-1, 0],
    mu2_is_t - 0.5 * x_best[:-1, 1]
])
z_is    = y[1:] - drift_is






# ──────────── OOS Rolling 序列粒子濾波 ───────────────────────────────────────────────


rv_start = np.array([
    np.mean(y[-RV_WIN-1:-1, 0]**2),
    np.mean(y[-RV_WIN-1:-1, 1]**2)
])                                        # IS/OOS 交接處的 RV 估計

# 加法平移（不是乘法縮放）：保留粒子雲的絕對離散度不變（跟 x_final 完全一樣），
# 只把平均水準搬到 rv_start，避免乘法縮放在 c 偏離 1 時壓縮/膨脹絕對多樣性。
# 下限 1e-10 防止平移後個別粒子跌到非正值（變異數必須為正）。
x_oos_init = np.maximum(x_final - x_final.mean(axis=0) + rv_start, 1e-10)

r2_oos       = np.random.randn(T_pred, N, 2)
x_oos        = x_oos_init.copy()
theta1_oos   = theta1_final.copy()
theta2_oos   = theta2_final.copy()        # 複製最後一期的粒子參數、狀態

x_oos_states = np.empty((T_pred, N, 2))
theta1_oos_states = np.empty((T_pred, N, k))
theta2_oos_states = np.empty((T_pred, N, k))     # 宣告估計值位置

h_roll       = np.empty((T_pred, 2))      # 滾動一步預測


# OOS迴圈(重抽樣、固定參數)
# OOS波動度估計值用 IS 最後一期的縮放當第一期
for t in range(T_pred):
    x_oos_states[t]      = x_oos
    theta1_oos_states[t] = theta1_oos
    theta2_oos_states[t] = theta2_oos     # 儲存估計值

    mu1 = x_oos[:, 0] + theta1_oos[:, 1] * (theta1_oos[:, 2] - x_oos[:, 0])
    mu2 = x_oos[:, 1] + theta2_oos[:, 1] * (theta2_oos[:, 2] - x_oos[:, 1])
    h_roll[t, 0] = float(np.mean(np.maximum(mu1, 1e-10)))
    h_roll[t, 1] = float(np.mean(np.maximum(mu2, 1e-10)))        # 儲存各時點一步預測的估計值

    if t < T_pred - 1:
        sx1  = np.sqrt(np.maximum(x_oos[:, 0], 1e-8))
        sx2  = np.sqrt(np.maximum(x_oos[:, 1], 1e-8))
        r1a  = (y_pred[t, 0] - (theta1_oos[:, 0] - 0.5 * x_oos[:, 0])) / sx1
        r1b  = (y_pred[t, 1] - (theta2_oos[:, 0] - 0.5 * x_oos[:, 1])) / sx2   # 殘差

        rho1 = theta1_oos[:, 4]
        rho2 = theta2_oos[:, 4]
        x_new = np.empty((N, 2))

        # 計算下期波動度
        x_new[:, 0] = np.maximum(
            x_oos[:, 0] + theta1_oos[:, 1] * (theta1_oos[:, 2] - x_oos[:, 0])
            + theta1_oos[:, 3] * sx1
              * (rho1 * r1a + np.sqrt(np.maximum(1 - rho1**2, 0)) * r2_oos[t, :, 0]),
            1e-10)
        x_new[:, 1] = np.maximum(
            x_oos[:, 1] + theta2_oos[:, 1] * (theta2_oos[:, 2] - x_oos[:, 1])
            + theta2_oos[:, 3] * sx2
              * (rho2 * r1b + np.sqrt(np.maximum(1 - rho2**2, 0)) * r2_oos[t, :, 1]),
            1e-10)

        # 重要性權重
        y_t      = np.tile(y_pred[t], (N, 1))
        MU_pred  = np.column_stack([theta1_oos[:, 0] - mu1 / 2, theta2_oos[:, 0] - mu2 / 2])
        SIG_pred = np.sqrt(np.column_stack([np.maximum(mu1, 1e-12), np.maximum(mu2, 1e-12)]))
        u_pred   = stats.norm.cdf(y_t, MU_pred, SIG_pred)
        yp_pred  = stats.norm.pdf(y_t, MU_pred, SIG_pred)
        cop_pred, _ = copulafitall23(copula_list, copula_list[best_ci_out],
                                      np.column_stack([u_pred[:, 0], u_pred[:, 1]]))
        lik_pred = cop_pred * yp_pred[:, 0] * yp_pred[:, 1]

        MU_new  = np.column_stack([theta1_oos[:, 0] - x_new[:, 0] / 2, theta2_oos[:, 0] - x_new[:, 1] / 2])
        SIG_new = np.sqrt(np.maximum(x_new, 1e-12))
        u_new   = stats.norm.cdf(y_t, MU_new, SIG_new)
        yp_new  = stats.norm.pdf(y_t, MU_new, SIG_new)
        cop_new, _ = copulafitall23(copula_list, copula_list[best_ci_out],
                                     np.column_stack([u_new[:, 0], u_new[:, 1]]))
        lik_new = cop_new * yp_new[:, 0] * yp_new[:, 1]

        w_oos = lik_new / np.maximum(lik_pred, 1e-300)
        w_sum = w_oos.sum()
        w_oos = w_oos / w_sum if w_sum > 0 else np.full(N, 1.0 / N)



        rs_oos     = np.random.choice(N, N, replace=True, p=w_oos)   # 重抽樣
        x_oos      = x_new[rs_oos]
        theta1_oos = theta1_oos[rs_oos]
        theta2_oos = theta2_oos[rs_oos]



mu_oos_t  = np.column_stack([
    np.mean(theta1_oos_states[:, :, 0], axis=1),
    np.mean(theta2_oos_states[:, :, 0], axis=1),
])                                                 # 每期粒子平均 mu
x_oos_mean = np.column_stack([
    np.mean(x_oos_states[:, :, 0], axis=1),
    np.mean(x_oos_states[:, :, 1], axis=1)
])                                                 # 每期粒子平均波動度
drift_oos = np.column_stack([
    mu_oos_t[:, 0] - 0.5 * x_oos_mean[:, 0],
    mu_oos_t[:, 1] - 0.5 * x_oos_mean[:, 1]
])                                                 # 漂移

z_pred    = y_pred - drift_oos                     # PIT





# ────────────── 以 RV 估計的真值與基準 ───────────────────────────────────────────────────────────

# 真值用來當 QLIKE/DM 的評分基準（事後量測），IS 接 OOS 後用置中窗估計
# 改用原始報酬平方（非模型漂移去中心化後的 z），避免 proxy／基準吃到模型自己估計出的漂移
y_concat  = np.concatenate([y[1:], y_pred], axis=0)
rv_concat = np.column_stack([
    pd.Series(y_concat[:, 0]**2).rolling(RV_WIN_TRUE, center=True).mean().values,
    pd.Series(y_concat[:, 1]**2).rolling(RV_WIN_TRUE, center=True).mean().values
])

n_is      = z_is.shape[0]
rv_is_m   = rv_concat[:n_is]   # IS的波動度真值
rv_pred_m = rv_concat[n_is:]   # OOS的波動度真值


# 基準：逐期向後窗 RV
# shift(2)：h_is[s]/h_roll[s] 預測波動度 s 時( y 的 t = s+1 )，公式使用的資訊到 s-1 期為止
bench_concat = np.column_stack([
    pd.Series(y_concat[:, 0]**2).rolling(RV_WIN).mean().shift(2).values,
    pd.Series(y_concat[:, 1]**2).rolling(RV_WIN).mean().shift(2).values
])

rv_bench_is  = np.maximum(bench_concat[:n_is], 1e-10)    # IS的波動度基準
rv_bench_oos = np.maximum(bench_concat[n_is:], 1e-10)    # OOS的波動度基準






# ────────── QLIKE 計算與比較表 ─────────────────────────────────────────

# 平均一步預測 QLIKE 計算 ( 公式 : log(估計) + 真值 / 估計 )
qlike_is        = np.nanmean(np.log(h_is[1:-1])   + rv_is_m[1:]   / h_is[1:-1],  axis=0)
qlike_oos       = np.nanmean(np.log(h_roll[:-1])  + rv_pred_m[1:] / h_roll[:-1], axis=0)
qlike_bench_is  = np.nanmean(np.log(rv_bench_is)  + rv_is_m   / rv_bench_is,  axis=0)
qlike_bench_oos = np.nanmean(np.log(rv_bench_oos) + rv_pred_m / rv_bench_oos, axis=0)


print(f'\n── QLIKE 比較（realized proxy = {RV_WIN_TRUE}-day centered y²（原始報酬平方，事後量測）──')
print(f'  {"":22s}  {"資產1":>12s}  {"資產2":>12s}')
print('  ' + '-' * 52)
print(f'  {"IS  SV-Copula":22s}  {qlike_is[0]:12.6f}  {qlike_is[1]:12.6f}')
print(f'  {"IS  基準(向後窗RV)":22s}  {qlike_bench_is[0]:12.6f}  {qlike_bench_is[1]:12.6f}')
print(f'  {"IS  差值(基準-模型)":22s}  {qlike_bench_is[0]-qlike_is[0]:12.6f}  {qlike_bench_is[1]-qlike_is[1]:12.6f}')
print('  ' + '-' * 52)
print(f'  {"OOS SV-Copula":22s}  {qlike_oos[0]:12.6f}  {qlike_oos[1]:12.6f}')
print(f'  {"OOS 基準(向後窗RV)":22s}  {qlike_bench_oos[0]:12.6f}  {qlike_bench_oos[1]:12.6f}')
print(f'  {"OOS 差值(基準-模型)":22s}  {qlike_bench_oos[0]-qlike_oos[0]:12.6f}  {qlike_bench_oos[1]-qlike_oos[1]:12.6f}')
print('  （差值 > 0 表示模型優於基準）')





# ────────── Rolling h-step Forecast QLIKE & DM 檢定（ OOS ）────────────────────────────────

print('\n── Rolling h-step Forecast QLIKE & DM 檢定（ 使用 CIR 解析解）──')
print(f'  {"h":>4s}  {"Q1":>12s}  {"DM1":>9s}  {"pl_1":>7s}'
      f'  {"Q2":>12s}  {"DM2":>9s}  {"pl_2":>7s}  {"n":>6s}')
print('  ' + '-' * 86)


FCST_HORIZONS = [1, 5, 10, 22]


# 根據 [1, 5, 10, 22] 天數計算 QLIKE
for h_fc in FCST_HORIZONS:
    n_periods = T_pred - h_fc
    if n_periods <= 0:
        continue

    st   = x_oos_states[:n_periods]
    k1   = theta1_oos_states[:n_periods, :, 1]
    th1  = theta1_oos_states[:n_periods, :, 2]
    k2   = theta2_oos_states[:n_periods, :, 1]
    th2  = theta2_oos_states[:n_periods, :, 2]

    # 多步預測公式
    fcast1 = th1 + (st[:, :, 0] - th1) * (1 - k1)**h_fc
    fcast2 = th2 + (st[:, :, 1] - th2) * (1 - k2)**h_fc
    fh1    = np.mean(np.maximum(fcast1, 1e-10), axis=1)
    fh2    = np.mean(np.maximum(fcast2, 1e-10), axis=1)

    # QLIKE 計算
    rv_tgt      = rv_pred_m[h_fc:h_fc+n_periods]
    bench_h     = rv_bench_oos[:n_periods]
    loss_sv_h1  = np.log(fh1) + rv_tgt[:, 0] / fh1
    loss_sv_h2  = np.log(fh2) + rv_tgt[:, 1] / fh2
    loss_bch_h1 = np.log(bench_h[:, 0]) + rv_tgt[:, 0] / bench_h[:, 0]
    loss_bch_h2 = np.log(bench_h[:, 1]) + rv_tgt[:, 1] / bench_h[:, 1]

    q1 = float(np.nanmean(loss_sv_h1))
    q2 = float(np.nanmean(loss_sv_h2))
    dm_h1, _, pl_h1 = _dm_test(loss_sv_h1, loss_bch_h1, h_horizon=h_fc)
    dm_h2, _, pl_h2 = _dm_test(loss_sv_h2, loss_bch_h2, h_horizon=h_fc)
    print(f'  {h_fc:>4d}  {q1:12.6f}  {dm_h1:9.4f}  {pl_h1:7.4f}{_sig(pl_h1)}'
          f'  {q2:12.6f}  {dm_h2:9.4f}  {pl_h2:7.4f}{_sig(pl_h2)}  {n_periods:>6d}')




# ────────── 各時點一步預測平均 QLIKE 計算 ( for 製圖 ) ────────────────────────────────

loss_sv_is     = np.log(h_is[1:-1])   + rv_is_m[1:]   / h_is[1:-1]
loss_bench_is  = np.log(rv_bench_is)  + rv_is_m   / rv_bench_is
loss_sv_oos    = np.log(h_roll[:-1])  + rv_pred_m[1:] / h_roll[:-1]
loss_bench_oos = np.log(rv_bench_oos) + rv_pred_m / rv_bench_oos      # 各時點的一步預測 QLIKE

loss_sv_is_t       = np.full((T - 1, 2), np.nan)
loss_sv_is_t[1:]   = loss_sv_is
loss_sv_oos_t       = np.full((T_pred, 2), np.nan)
loss_sv_oos_t[1:]   = loss_sv_oos                          # 適當調整位置



# ────────── 製圖 ────────────────────────────────────────────────────────────────────────────

# 圖：IS
t_is_arr      = np.arange(1, T)
_proxy_lbl_is = f'y² proxy (centered {RV_WIN_TRUE}d)' if RV_WIN_TRUE > 1 else 'y² proxy (逐期)'

fig_is, axes_is = plt.subplots(2, 2, figsize=(14, 8))
for ai, aname in enumerate(['資產1', '資產2']):
    axes_is[0, ai].plot(t_is_arr,      rv_is_m[:, ai],   label=_proxy_lbl_is,              alpha=0.6)
    # h_is[1:-1][s]（＝h_is[s+1]）預測的是目標期 s+2，畫在 t_is_arr[1:]（=[2,...,T-1]）才對齊
    axes_is[0, ai].plot(t_is_arr[1:], h_is[1:-1, ai], label='IS 1-step forecast',       linewidth=1.5)
    axes_is[0, ai].set_title(f'IS 1-step Forecast vs y² Proxy - {aname}')
    axes_is[0, ai].legend()
    axes_is[1, ai].plot(t_is_arr, loss_sv_is_t[:, ai],   label='SV-Copula')
    axes_is[1, ai].plot(t_is_arr, loss_bench_is[:, ai],  label='基準(向後窗RV)', alpha=0.5)
    axes_is[1, ai].set_title(f'IS QLIKE - {aname}')
    axes_is[1, ai].set_xlabel('t (IS)')
    axes_is[1, ai].legend()
plt.suptitle('In-Sample Volatility Forecast & QLIKE', fontsize=13)
plt.tight_layout()
fig_is.savefig('is_forecast_qlike.png', dpi=150, bbox_inches='tight')
plt.show()


# 圖：OOS
t_pred_arr = np.arange(T_pred)
_proxy_lbl = f'y² proxy (centered {RV_WIN_TRUE}d)' if RV_WIN_TRUE > 1 else 'y² proxy (逐期)'

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8))
for ai, aname in enumerate(['資產1', '資產2']):
    axes3[0, ai].plot(t_pred_arr,    rv_pred_m[:, ai], label=_proxy_lbl,             alpha=0.6)
    axes3[0, ai].plot(t_pred_arr[1:], h_roll[:-1, ai], label='Rolling 1-step forecast', linewidth=1.5)
    axes3[0, ai].set_title(f'OOS 1-step Forecast vs y² Proxy - {aname}')
    axes3[0, ai].legend()
    axes3[1, ai].plot(t_pred_arr, loss_sv_oos_t[:, ai],   label='SV-Copula')
    axes3[1, ai].plot(t_pred_arr, loss_bench_oos[:, ai],  label='基準(向後窗RV)', alpha=0.5)
    axes3[1, ai].set_title(f'OOS QLIKE - {aname}')
    axes3[1, ai].set_xlabel('t (OOS)')
    axes3[1, ai].legend()
plt.tight_layout()
fig3.savefig('oos_forecast_qlike.png', dpi=150, bbox_inches='tight')
plt.show()








# ─────────────────────────────────────────────────────────────────────────────
# ── VaR 回測（OOS）：Kupiec POF & Christoffersen CC ──────────────────────────
# z_alpha 由 IS 殘差估計（無前視），條件 VaR 來自 OOS 粒子濾波 1-step 預測
# ─────────────────────────────────────────────────────────────────────────────

SEP         = '  '
mu_hat_oos  = drift_oos
sig_hat_oos = np.sqrt(np.maximum(x_oos_mean, 1e-10))   # 與 drift_oos 同源（x_oos_states），非 h_roll，避免均值/波動用不同期變異數

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

        row = [
            _lj(lab,                       WCOL[0]),
            _rj(f'{n_viol}/{T_pred}',      WCOL[1]),
            _rj(f'{p_hat:.4f}',            WCOL[2]),
            _rj(_fl(lr_uc),                WCOL[3]),
            _rj(_fl(p_uc)  + _sig(p_uc),   WCOL[4]),
            _rj(_fl(lr_ind),               WCOL[5]),
            _rj(_fl(p_ind) + _sig(p_ind),  WCOL[6]),
        ]
        print('    ' + SEP.join(row))




# ─────────────────────────────────────────────────────────────────────────────
# ──  Rosenblatt OOS PIT：i.i.d. U(0,1) 檢定（無 look-ahead）─────────────
# ─────────────────────────────────────────────────────────────────────────────

# OOS 邊際 PIT（drift_oos[t] 基於觀察 y_pred[t] 前的粒子狀態；波動度改用置中窗 RV 真值，見上）
sig_oos_pit = np.sqrt(rv_pred_m)                            # (T_pred, 2)  ← RV 置中窗真值
u_oos_pit   = stats.norm.cdf((y_pred - drift_oos) / sig_oos_pit)
u_oos_pit   = np.clip(u_oos_pit, 1e-10, 1 - 1e-10)
valid_oos   = ~np.any(np.isnan(u_oos_pit), axis=1)   # 置中窗右側缺未來資料，OOS 尾端排除
#
# IS copula 固定擬合（只用 IS 邊際 PIT u，OOS 評估時無前視）
_FAMILY_MAP_PD = {
    'copulaN': pv.BicopFamily.gaussian,
    'copulaT': pv.BicopFamily.student,
    'copulaC': pv.BicopFamily.clayton,
    'copulaF': pv.BicopFamily.frank,
    'copulaG': pv.BicopFamily.gumbel,
}
# bc_is = pv.Bicop(family=_FAMILY_MAP_PD[copula_list[best_ci_out]])
# bc_is.fit(np.clip(u, 1e-6, 1 - 1e-6))
#
# # 聯合 Rosenblatt 轉換：z₁ = u₁，z₂ = C_{2|1}(u₂|u₁; IS copula)
# z_rosen = np.full((T_pred, 2), np.nan)
# z_rosen[valid_oos, 0] = u_oos_pit[valid_oos, 0]
# z_rosen[valid_oos, 1] = bc_is.hfunc1(u_oos_pit[valid_oos]).ravel()
# z_rosen[valid_oos]    = np.clip(z_rosen[valid_oos], 1e-10, 1 - 1e-10)
#
#
# def _rj_tag(p):
#     return '  （拒絕）' if p < 0.05 else '  （未拒絕）'
#
#
# def _print_pit_block(u_mat, var_labels, title):
#     """列印一組 Rosenblatt PIT 的 LB 檢定結果。"""
#     print(f'\n  ── {title} ──')
#     for ai, lbl in enumerate(var_labels):
#         col = u_mat[:, ai]
#         col = col[~np.isnan(col)]
#         v   = stats.norm.ppf(col)
#         lbm = float(acorr_ljungbox(v,    lags=[20], return_df=True)['lb_pvalue'].iloc[-1])
#         lbv = float(acorr_ljungbox(v**2, lags=[20], return_df=True)['lb_pvalue'].iloc[-1])
#         print(f'\n    {lbl}：')
#         print(f'      LB(20)  Φ⁻¹(u) 串列:     p={lbm:.4f}{_rj_tag(lbm)}')
#         print(f'      LB²(20) [Φ⁻¹(u)]² ARCH:  p={lbv:.4f}{_rj_tag(lbv)}')
#
#
# print(f'\n── Rosenblatt OOS PIT 檢定（無 look-ahead，IS 擬合 {copula_list[best_ci_out]} copula）──')
# print(f'  OOS 邊際波動度：滯後滾動 RV（RV_WIN={RV_WIN}，僅用 t-1 以前資料，前 {RV_WIN} 期無值已排除）')
# print('  H₀：序列為 i.i.d. U(0,1)；p < 0.05 拒絕 H₀')
#
# _print_pit_block(z_rosen,   ['z₁ = u₁（邊際）', 'z₂ = C₂|₁(u₂|u₁)（條件）'],
#                  '聯合 Rosenblatt U(0,1) 檢定（OOS，文獻標準）')




# ─────────────────────────────────────────────────────────────────────────────
# ── Copula-only OOS DM：各 Copula 增益 vs 獨立基準 ───────────────────────
# ─────────────────────────────────────────────────────────────────────────────

print('\n── Copula-only OOS DM（log c(u₁,u₂) vs 獨立基準）──')
print('  各 copula 以 IS 邊際 PIT 固定擬合後評估 OOS log c（copula 擬合本身無前視；')
print('  邊際 PIT 改用置中窗 RV 真值，屬事後量測，非嚴格前向）')
print('  損失差 dₜ = −log c(u₁ₜ,u₂ₜ)；H₁(gain): E[d]<0 ⟺ copula 優於獨立（* p<0.05）')
print(f'\n  {"Copula":12s}  {"avg log c":>12s}  {"DM stat":>10s}  {"p(gain)":>8s}')
print('  ' + '-' * 48)

u_oos_c = np.clip(u_oos_pit, 1e-6, 1 - 1e-6)
lc_all  = np.full((T_pred, n_cop), np.nan)   # 逐日 log c(u₁,u₂)，5 個 copula 都存
for ci, cop_name in enumerate(copula_list):
    try:
        bc_c = pv.Bicop(family=_FAMILY_MAP_PD[cop_name])
        bc_c.fit(np.clip(u, 1e-6, 1 - 1e-6))
        lc   = np.full(T_pred, np.nan)
        lc[valid_oos] = np.clip(
            np.log(np.maximum(bc_c.pdf(u_oos_c[valid_oos]), 1e-300)), -50.0, 50.0)
    except Exception:
        lc   = np.zeros(T_pred)
    lc_all[:, ci] = lc
    avg_lc        = float(np.nanmean(lc))
    dm_s, _, pl   = _dm_test(-lc, np.zeros(T_pred), h_horizon=1)
    note = ' ← 最佳' if ci == best_ci_out else ''
    print(f'  {cop_name:12s}  {avg_lc:12.6f}  {dm_s:10.4f}  {pl:8.4f}{_sig(pl)}{note}')

print('\n  （avg log c > 0：OOS 聯合密度優於獨立；p(gain) < 0.05：copula 增益統計顯著）')



# ─────────────────────────────────────────────────────────────────────────────
# ─────────────── OOS 逐日最佳 Copula ───────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 邏輯：每個 copula 都用同一組 IS 邊際 PIT 固定擬合（無前視，見上），
# 每一天各自比較 lc_all（log c）argmax，不做滾動視窗加總；
# 連續判為同一個 copula 的日子合併列印成一段，方便閱讀。

valid_day = ~np.any(np.isnan(lc_all), axis=1)
best_day  = np.full(T_pred, -1, dtype=int)
best_day[valid_day] = np.argmax(lc_all[valid_day], axis=1)

fig_cop_day, ax_cop_day = plt.subplots(figsize=(14, 2.2))
_colors = {'copulaN': '#888888', 'copulaT': '#1f77b4', 'copulaC': '#2ca02c',
           'copulaF': '#d62728', 'copulaG': '#9467bd'}
for ci, cop_name in enumerate(copula_list):
    mask = valid_day & (best_day == ci)
    ax_cop_day.fill_between(np.arange(T_pred), 0, 1, where=mask,
                             color=_colors[cop_name], step='mid', label=cop_name)
ax_cop_day.set_yticks([])
ax_cop_day.set_xlabel('t (OOS)')
ax_cop_day.set_title('OOS 逐日最佳 Copula')
ax_cop_day.legend(loc='upper right', ncol=5, fontsize=8)
plt.tight_layout()
fig_cop_day.savefig('oos_daily_best_copula.png', dpi=150, bbox_inches='tight')
plt.show()









# ═══════════════════════════════════════════════════════════════════════════
# ────────────── OOS 初期短期 τ 追蹤診斷 ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════



SHORT_WIN = RV_WIN             # 10，短期尺度，與 RV_WIN 對齊（τ_model／RW基準的回望窗長度）
TRUTH_WIN = 2 * SHORT_WIN      # 20，置中窗長度（真值估計專用，跟 SHORT_WIN 脫鉤，可獨立調大）
_half_truth = TRUTH_WIN // 2

# _early_mask      = valid_day.copy(); _early_mask[SHORT_WIN:]      = False
# _early2_mask     = valid_day.copy(); _early2_mask[:SHORT_WIN] = False; _early2_mask[2 * SHORT_WIN:] = False
# _n_early_best    = int(np.sum(_early_mask  & (best_day == best_ci_out)))
# _n_early2_best   = int(np.sum(_early2_mask & (best_day == best_ci_out)))
# print(f'\n  主迴圈最佳 Copula（{copula_list[best_ci_out]}）在初期（第1~{SHORT_WIN}天）'
#       f'為當日最優的天數：{_n_early_best} / {int(_early_mask.sum())}')
# print(f'  主迴圈最佳 Copula（{copula_list[best_ci_out]}）在次初期（第{SHORT_WIN + 1}~{2 * SHORT_WIN}天）'
#       f'為當日最優的天數：{_n_early2_best} / {int(_early2_mask.sum())}')

print('\n' + '═'*78)
print(f' OOS 初期短期 τ 追蹤診斷（SHORT_WIN={SHORT_WIN}，IS 尾端墊底回望）')
print('═'*78)
print(f'  τ_model 短窗版：回望 {SHORT_WIN} 期（嚴格 t 之前，不含 t，無前視），')
print(f'  前 {SHORT_WIN} 期用 IS 尾端 RV-PIT 墊底，讓 OOS 從第1期就有值')
print(f'  邊際：與 u_oos_c 同一套「滯後滾動 RV（RV_WIN={RV_WIN}）」，避免原始/PIT空間錯配')
print(f'  τ_realized（真值）：置中窗 [t-{_half_truth}, t+{_half_truth})，長度TRUTH_WIN={TRUTH_WIN}，')
print(f'  允許使用 t 前後資料，目的是降低真值估計本身的雜訊，非嚴格前向預測評估')
print(f'  （置中窗跟 τ_model 的回望窗長度脫鉤，詳見下方對照 RW/h_roll 基準）')
print()

def _mae_bias(e):
    e = e[~np.isnan(e)]
    if len(e) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(np.abs(e))), float(np.mean(e)), len(e)  # mae,bias計算函式







# ──────────────  計算 tau ( 模型 copula 估計、 stats.kendalltau的估計跟真值 ) ─────────────────────


# 模型部分
_pad   = max(SHORT_WIN, _half_truth)
_warm  = _pad + RV_WIN + 5                       # 抓 IS 尾端，確保 rolling(RV_WIN) 初期有效

_z_br  = np.concatenate([z_is[-_warm:], z_pred], axis=0)
_rv_br = np.column_stack([
    pd.Series(_z_br[:, 0]**2).rolling(RV_WIN).mean().shift(2).values,
    pd.Series(_z_br[:, 1]**2).rolling(RV_WIN).mean().shift(2).values
])                                               # 使用在 QLIKE 計算的去漂移殘差估計 RV

_rv_br = np.maximum(_rv_br, 1e-10)
_u_br  = np.clip(stats.norm.cdf(_z_br / np.sqrt(_rv_br)), 1e-6, 1 - 1e-6)
_u_br  = _u_br[_warm - _pad:]
_oos0  = _pad                                   # 以 RV 估計的波動度計算 PIT

_tau_mod_short = np.full(T_pred, np.nan)        # 計算模型估計的 tau ( 波動度與 RV估計一致版 )
for _t in range(T_pred):
    _win = _u_br[_oos0 + _t - SHORT_WIN: _oos0 + _t]
    if len(_win) < SHORT_WIN or np.any(np.isnan(_win)):
        continue
    _, _tau_s = copulafitall23(copula_list, copula_list[best_ci_out], _win)
    _tau_mod_short[_t] = _tau_s



# 真值 ( 報酬率置中窗 RV 估計 )
_pad_y  = max(SHORT_WIN, _half_truth)
_y_br   = np.concatenate([y[-_pad_y:], y_pred], axis=0)    # 原始報酬橋接（IS尾端+OOS）
_oos0_y = _pad_y

_tau_real_short = np.full(T_pred, np.nan)      # tau 真值：置中窗 [t-_half_truth, t+_half_truth)
for _t in range(0, T_pred - _half_truth):
    _win_y = _y_br[_oos0_y + _t - _half_truth: _oos0_y + _t - _half_truth + TRUTH_WIN]
    _kt, _ = stats.kendalltau(_win_y[:, 0], _win_y[:, 1])
    _tau_real_short[_t] = _kt



# 天真基準 ( 報酬率置後窗 RV 估計 )
_tau_rw_short = np.full(T_pred, np.nan)        # 比較用 tau 估計：置前窗
for _t in range(T_pred):
    _win_y = _y_br[_oos0_y + _t - SHORT_WIN: _oos0_y + _t]
    if len(_win_y) < SHORT_WIN:
        continue
    _kt, _ = stats.kendalltau(_win_y[:, 0], _win_y[:, 1])
    _tau_rw_short[_t] = _kt







# ────────────── 結果表格 ──────────────────────────────────────────────────────────────────────


_seg_bounds = [(0, SHORT_WIN, f'OOS 第 1~{SHORT_WIN} 期（初期）'),
               (SHORT_WIN, 2 * SHORT_WIN, f'OOS 第 {SHORT_WIN + 1}~{2 * SHORT_WIN} 期（次初期）'),
               (2 * SHORT_WIN, T_pred - _half_truth, f'OOS 第 {2 * SHORT_WIN + 1} 期起（穩態段）')]

_SW = [30, 8, 4, 9, 9, 10, 10]   # 區段/space/n/RV模型MAE/RW基準MAE/RV模型bias/RW基準bias
_SH = ['區段', 'space', 'n', 'RV模型MAE', 'RW基準MAE', 'RV模型bias', 'RW基準bias']
print('  ' + '  '.join(_lj(h, w) if i < 2 else _rj(h, w) for i, (h, w) in enumerate(zip(_SH, _SW))))
print('  ' + '-' * (sum(_SW) + 2 * (len(_SW) - 1)))
for _lo, _hi, _lbl in _seg_bounds:
    _hi = min(_hi, T_pred - _half_truth)
    if _hi <= _lo:
        print('  ' + _lj(f'{_lbl}：樣本不足', _SW[0]))
        continue
    _sl = slice(_lo, _hi)

    _mae_m,  _bias_m,  _n  = _mae_bias(_tau_real_short[_sl] - _tau_mod_short[_sl])
    _mae_rw, _bias_rw, _   = _mae_bias(_tau_real_short[_sl] - _tau_rw_short[_sl])
    print('  ' + '  '.join([
        _lj(_lbl, _SW[0]), _lj('原始報酬', _SW[1]), _rj(_n, _SW[2]),
        _rj(f'{_mae_m:.4f}', _SW[3]), _rj(f'{_mae_rw:.4f}', _SW[4]),
        _rj(f'{_bias_m:.4f}', _SW[5]), _rj(f'{_bias_rw:.4f}', _SW[6]),
    ]))
print()
print('  判準：模型MAE 明顯小於RW基準MAE → copula 擬合本身有額外貢獻，不只是持續性；')
print('        初期 MAE 明顯大於穩態段 MAE → 暖機證據；相當 → IS/OOS 交接乾淨')
print()

# ── 圖：RV模型估計 τ vs RV真值 τ（OOS，逐期）──────────────────────────
fig_tau_rv, ax_tau_rv = plt.subplots(figsize=(14, 4))
ax_tau_rv.plot(_tau_mod_short, label='τ_model（RV模型估計）', lw=1.0)
ax_tau_rv.plot(_tau_real_short, label='τ_realized（RV真值，置中窗）', lw=1.0)
ax_tau_rv.axvline(SHORT_WIN, color='gray', lw=0.6, linestyle=':')
ax_tau_rv.axvline(2 * SHORT_WIN, color='gray', lw=0.6, linestyle=':')
ax_tau_rv.set_title(f'OOS 短窗 Kendall τ：RV模型估計 vs RV真值（SHORT_WIN={SHORT_WIN}）')
ax_tau_rv.set_xlabel('OOS 時間索引'); ax_tau_rv.set_ylabel('Kendall τ')
ax_tau_rv.legend()
plt.tight_layout()
fig_tau_rv.savefig('oos_tau_model_vs_real.png', dpi=150, bbox_inches='tight')
plt.show()







# ═══════════════════════════════════════════════════════════════════════════
# ── 對照組：τ_model 改用 x_oos_mean（粒子濾波器自身「當期」波動度後驗）─────────
# 目的：量化「用模型自己即時濾波出的波動度」相對於「用外部RV」的追蹤表現落差。
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '═'*78)
print(f' 對照：τ_model 改用 x_oos_mean（模型自身波動度後驗，IS 尾端用 x_best 當期後驗墊底）')
print('═'*78)
print('  ⚠ 初期區段用 x_best 當期後驗（mu_hat/sig_hat）墊底，可能混有資訊洩漏，結果僅供參考')
print('    （次初期／穩態段純落在 OOS，不受此限制）')
print()


_pad_h      = max(SHORT_WIN, _half_truth)

_u_oos_h    = np.clip(stats.norm.cdf(z_pred / sig_hat_oos), 1e-6, 1 - 1e-6)
_u_is_h     = np.clip(u[T - _pad_h:T], 1e-6, 1 - 1e-6)         # 拿以前算過東西的來計算 PIT

_u_oos_h_br = np.concatenate([_u_is_h, _u_oos_h], axis=0)     # 橋接
_oos0_h     = _pad_h


_tau_mod_short_h = np.full(T_pred, np.nan)        # 以模型估計波動度計算的 tau
for _t in range(T_pred):
    _win = _u_oos_h_br[_oos0_h + _t - SHORT_WIN: _oos0_h + _t]
    if len(_win) < SHORT_WIN or np.any(np.isnan(_win)):
        continue
    _, _tau_h = copulafitall23(copula_list, copula_list[best_ci_out], _win)
    _tau_mod_short_h[_t] = _tau_h


# 表格（n(RV) 跟 n(xm) 分開顯示，避免用不同覆蓋率的樣本互相比較 MAE）
_SW2 = [30, 8, 5, 5, 10, 12, 10, 12]
_SH2 = ['區段', 'space', 'n(RV)', 'n(xm)', 'RV模型MAE', 'x_oos_mean模型MAE', 'RV模型bias', 'x_oos_mean模型bias']
print('  ' + '  '.join(_lj(h, w) if i < 2 else _rj(h, w) for i, (h, w) in enumerate(zip(_SH2, _SW2))))
print('  ' + '-' * (sum(_SW2) + 2 * (len(_SW2) - 1)))
for _lo, _hi, _lbl in _seg_bounds:
    _hi = min(_hi, T_pred - _half_truth)
    if _hi <= _lo:
        print('  ' + _lj(f'{_lbl}：樣本不足', _SW2[0]))
        continue
    _sl = slice(_lo, _hi)

    _mae_rv, _bias_rv, _n  = _mae_bias(_tau_real_short[_sl] - _tau_mod_short[_sl])
    _mae_h,  _bias_h,  _nh = _mae_bias(_tau_real_short[_sl] - _tau_mod_short_h[_sl])
    _mae_h_s  = f'{_mae_h:.4f}'  if _nh > 0 else 'n/a'
    _bias_h_s = f'{_bias_h:.4f}' if _nh > 0 else 'n/a'
    _cov_warn = '  ← n(xm)<n(RV)，x_oos_mean版樣本覆蓋率較低，MAE比較需謹慎' if _nh < _n else ''
    print('  ' + '  '.join([
        _lj(_lbl, _SW2[0]), _lj('原始報酬', _SW2[1]), _rj(_n, _SW2[2]), _rj(_nh, _SW2[3]),
        _rj(f'{_mae_rv:.4f}', _SW2[4]), _rj(_mae_h_s, _SW2[5]),
        _rj(f'{_bias_rv:.4f}', _SW2[6]), _rj(_bias_h_s, _SW2[7]),
    ]) + _cov_warn)



print()
print('  判準：x_oos_mean版 MAE 明顯大於 RV版 → 邊際波動度預測品質正在拖累追蹤表現；')
print('        兩者接近 → 目前邊際預測已經夠好，不是主要瓶頸')
print('        （比較前請先確認 n(RV) 與 n(xm) 是否接近，落差過大代表比較基礎不公平）')
print('  ※ 初期區段(x_oos_mean版)為 x_best 當期後驗墊底結果，可能有資訊洩漏，解讀需謹慎（見上方註解）')
print()










# ═══════════════════════════════════════════════════════════════════════════
# ──────────  FKO ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════




# ──────────────── FKO 共用參數與輔助函式 ──────────────────────────────────────────────

from scipy.optimize import brentq as _brentq

RF_DAILY = 0.0                # 每日無風險利率
TC_BPS   = 10.0 / 1e4         # 成本率
GAMMAS   = [1.0, 5.0, 10.0]   # 風險係數
ANN      = 252                # 一年天數


# 相關性矩陣 ( rho 、 波動度 )
def _build_cov(rho_series, h_mat):
    T_  = len(rho_series)
    cov = np.zeros((T_, 2, 2))
    cov[:, 0, 0] = h_mat[:, 0]
    cov[:, 1, 1] = h_mat[:, 1]
    cov[:, 0, 1] = rho_series * np.sqrt(h_mat[:, 0] * h_mat[:, 1])
    cov[:, 1, 0] = cov[:, 0, 1]
    return cov


# 檢查相關性矩陣 SPD ( 某時點矩陣 )
def _spd_inv(Sig, min_eig=1e-8):
    vals, vecs = np.linalg.eigh(Sig)
    vals = np.maximum(vals, min_eig)
    return vecs @ np.diag(1.0 / vals) @ vecs.T


# 求「 實現組合報酬 、 週轉率 、 權重序列 」 ( 相關性矩陣 、 報酬率 、 風險係數 )
def _portfolio_ra(cov_seq, mu_t, gamma):
    Rp_arr = np.zeros(T_pred)
    to_arr = np.zeros(T_pred)
    w_seq  = np.zeros((T_pred, 2))
    w_prev = np.zeros(2)
    for t in range(T_pred):
        mu_ex = mu_t[t] - RF_DAILY
        Sinv  = _spd_inv(cov_seq[t])
        w_r   = (1.0 / gamma) * (Sinv @ mu_ex)
        w_seq[t]  = w_r
        to_arr[t] = float(np.sum(np.abs(w_r - w_prev)))
        w_prev    = w_r.copy()
        Rp_arr[t] = float(w_r @ y_pred[t]) + (1.0 - float(np.sum(w_r))) * RF_DAILY
    return Rp_arr, to_arr, w_seq


# 二次效用逐期實現值 ( 把「組合報酬」轉換成「投資人主觀滿意度」 ) ( 實現組合報酬序列 、 風險報酬 )
def _utility(Rp, gamma):
    g = gamma / (1.0 + gamma)
    return (1.0 + Rp) - 0.5 * g * (1.0 + Rp)**2


# 績效費 ( 組合報酬 、 風險係數)
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
    return np.nan


# nan處理
def _ff(v, fmt='.4f'):
    return format(v, fmt) if not np.isnan(v) else 'nan'







# ──────────────────── 績效費計算、檢定、表格 ─────────────────────────────────────────────────


VAL_WIN = 22   # FKO 報酬率驗證窗口：OOS 初期一個月（22 個交易日）

_FW3 = [4, 14, 7, 11, 11, 9]
_FH3 = ['γ', '模型', '含TC', 'Δ(bps)', 'DM-stat', 'p(gain)']



# 拿計算過的波動度、tau 來用
_rv_true   = pd.DataFrame(rv_pred_m).ffill().values
_rv_true   = np.maximum(_rv_true, 1e-10)

_tau2f     = float(R_mat[0, best_ci_out])                                        # IS的最終估計 tau
_rho2f     = float(np.clip(np.sin(np.pi * _tau2f / 2.0), -0.999, 0.999))
rho_series = np.clip(np.sin(np.pi * _tau_mod_short / 2.0), -0.999, 0.999)  # 以公式估計 rho
rho_series = np.where(np.isnan(rho_series), _rho2f, rho_series)                  # 保險 fallback

_mu_is_hist = np.mean(y, axis=0)   # IS 各資產日報酬均值
_dr_rv      = np.column_stack([
    np.full(T_pred, _mu_is_hist[0]) - _rv_true[:, 0] / 2,
    np.full(T_pred, _mu_is_hist[1]) - _rv_true[:, 1] / 2
])                                 # 報酬率均值 ( implies 共用相同 μ_ex )



# 模型 tau 的實現組合報酬 ( SV-Copula 與基準 SepSV  )
_cov_c2_rv = _build_cov(rho_series,       _rv_true)
_cov_s2_rv = _build_cov(np.zeros(T_pred), _rv_true)

_Rp_c2_rv = {}; _To_c2_rv = {}
_Rp_s2_rv = {}; _To_s2_rv = {}
for _gm in GAMMAS:
    _Rp_c2_rv[_gm], _To_c2_rv[_gm], _ = _portfolio_ra(_cov_c2_rv, _dr_rv, _gm)
    _Rp_s2_rv[_gm], _To_s2_rv[_gm], _ = _portfolio_ra(_cov_s2_rv, _dr_rv, _gm)



# 真值 tau：OOS 置中窗 Kendall 的實現組合報酬
_rho_real = np.clip(np.sin(np.pi * _tau_real_short / 2.0), -0.999, 0.999)
_rho_real = np.where(np.isnan(_rho_real), _rho2f, _rho_real)
_cov_real_rv = _build_cov(_rho_real, _rv_true)

_Rp_real_rv  = {}; _To_real_rv = {}
for _gm in GAMMAS:
    _Rp_real_rv[_gm], _To_real_rv[_gm], _ = _portfolio_ra(_cov_real_rv, _dr_rv, _gm)



# 固定 tau：IS 估計的實現組合報酬
_cov_fixed_rv = _build_cov(np.full(T_pred, _rho2f), _rv_true)

_Rp_fixed_rv = {}; _To_fixed_rv = {}
for _gm in GAMMAS:
    _Rp_fixed_rv[_gm], _To_fixed_rv[_gm], _ = _portfolio_ra(_cov_fixed_rv, _dr_rv, _gm)




print('\n' + '═'*78)
print(f' FKO（oracle RV + IS 歷史均值）— 差異純為 copula ρ（僅 OOS 前 {VAL_WIN} 天）')
print('═'*78)
print(f'  Σ 對角線與 Itô 修正：rv_pred_m（{RV_WIN_TRUE} 期置中窗 oracle RV）')
print(f'  μ̂_t：IS 期樣本均值（固定，資產1={_mu_is_hist[0]:.6f}，資產2={_mu_is_hist[1]:.6f}）')
print('  三模型 μ_ex 與 Σ 對角線完全相同，SV-Copula vs SepSV 差異純為 copula ρ')
print('  ！波動度含未來資訊（oracle），為不可行估計，僅供隔離 ρ 效果參考')
print(f'  驗證窗口：僅取 OOS 前 {VAL_WIN} 個交易日，與表二一致')
print()
print('  ' + '  '.join(_rj(h, w) for h, w in zip(_FH3, _FW3)))
print('  ' + '  '.join('-'*w for w in _FW3))

_pc3_rv = {_gm: {
    'SV-Copula':   (_Rp_c2_rv[_gm], _To_c2_rv[_gm]),
    'Separate SV': (_Rp_s2_rv[_gm], _To_s2_rv[_gm]),
    '真值τ':        (_Rp_real_rv[_gm],  _To_real_rv[_gm]),
    '固定τ':        (_Rp_fixed_rv[_gm], _To_fixed_rv[_gm]),
} for _gm in GAMMAS}


for _gm3 in GAMMAS:

    _Rb3r_rv, _Tb3r_rv = _pc3_rv[_gm3]['Separate SV']

    for _mn3 in ['SV-Copula', '真值τ', '固定τ']:

        _Rm3r_rv, _Tm3r_rv = _pc3_rv[_gm3][_mn3]

        for _tc3, _tcl3 in [(False, '否'), (True, '是')]:               # ( 交易成本開關是 _tc3 )

            # 模型與基準的實現組合報酬 - 交易成本 ( TC_BPS * _Tm3r_rv 是 成本率 × 週轉率)
            _Rm3_rv = (_Rm3r_rv - TC_BPS * _Tm3r_rv if _tc3 else _Rm3r_rv.copy())[:VAL_WIN]
            _Rb3_rv = (_Rb3r_rv - TC_BPS * _Tb3r_rv if _tc3 else _Rb3r_rv.copy())[:VAL_WIN]

            _dd3_rv   = _perf_fee(_Rm3_rv, _Rb3_rv, _gm3)                         # 績效費
            _dbps3_rv = _dd3_rv * ANN * 1e4 if not np.isnan(_dd3_rv) else np.nan  # 績效費年化

            _Um3_rv   = _utility(_Rm3_rv, _gm3)
            _Ub3_rv   = _utility(_Rb3_rv, _gm3)
            _ds3_rv, _, _pg3_less_rv = _dm_test(-_Ub3_rv, -_Um3_rv)
            _pg3_rv = 1.0 - _pg3_less_rv                                          # 績效費檢定

            print('  ' + '  '.join([
                _rj(f'{_gm3:.0f}',          _FW3[0]),
                _lj(_mn3,                      _FW3[1]),
                _rj(_tcl3,                     _FW3[2]),
                _rj(_ff(_dbps3_rv, '.2f'), _FW3[3]),
                _rj(_ff(_ds3_rv),              _FW3[4]),
                _rj(_ff(_pg3_rv) + (_sig(_pg3_rv) if not np.isnan(_pg3_rv) else ''), _FW3[5]),
            ]))
    print()







# ──────────────────── OOS 每日累積報酬差折線圖 ─────────────────────────────────────────────────


_days_x = np.arange(1, VAL_WIN + 1)

def _cum_diff_bps(Rp_m, Rp_b):
    return np.cumsum((Rp_m - Rp_b)[:VAL_WIN]) * 1e4

# ── oracle RV + IS 均值（模型τ vs ρ=0；真值τ vs ρ=0；固定τ vs ρ=0，各一張）─────
_oracle_d2_cfgs = [
    ('模型τ', _Rp_c2_rv,    _To_c2_rv,    '#2196F3', 'fko_daily_oracle_model.png'),
    ('真值τ', _Rp_real_rv,  _To_real_rv,  '#4CAF50',  'fko_daily_oracle_true.png'),
    ('固定τ', _Rp_fixed_rv, _To_fixed_rv, '#FF9800',  'fko_daily_oracle_fixed.png'),
]

for _oname, _Rp_o, _To_o, _clr, _fname in _oracle_d2_cfgs:
    _fig6, _axes6 = plt.subplots(len(GAMMAS), 1,
                                  figsize=(7, 4 * len(GAMMAS)), sharey=False)
    if len(GAMMAS) == 1:
        _axes6 = [_axes6]
    for _gi, _gm in enumerate(GAMMAS):
        _ax6 = _axes6[_gi]
        _Rb_d0 = _Rp_s2_rv[_gm]; _Tb_d0 = _To_s2_rv[_gm]
        for _tc, _tcl, _ls in [(False, '不含TC', '-'), (True, '含TC', '--')]:
            _Rm_d = _Rp_o[_gm] - TC_BPS * _To_o[_gm] if _tc else _Rp_o[_gm]
            _Rb_d = _Rb_d0 - TC_BPS * _Tb_d0 if _tc else _Rb_d0
            _ax6.plot(_days_x, _cum_diff_bps(_Rm_d, _Rb_d),
                      label=_tcl, color=_clr, linewidth=1.0, linestyle=_ls)
        _ax6.axhline(0, color='black', linewidth=0.8, linestyle=':')
        _ax6.set_title(f'γ={_gm:.0f}', fontsize=10)
        _ax6.set_xlabel('OOS 日')
        if _gi == 0:
            _ax6.set_ylabel('累積報酬差（bps）')
        _ax6.legend(fontsize=8)
    _fig6.suptitle(f'OOS 每日累積報酬差（oracle RV，{_oname} vs ρ=0，前{VAL_WIN}天）',
                   fontsize=12)
    _fig6.tight_layout()
    _fig6.savefig(_fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  [{_oname} 每日折線圖已儲存至 {_fname}]')


