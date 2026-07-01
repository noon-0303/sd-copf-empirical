import sys
import unicodedata
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import warnings

from Empirical.copulafitall_sd_2 import copulafitall3

warnings.filterwarnings('ignore')
# np.random.seed(0)

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

COPULA_NAMES  = ['N', 'T', 'C', 'F', 'G']
n_cop         = len(COPULA_NAMES)

FCST_HORIZONS = [1, 5, 10, 22]
WIN_OOS_MAX   = 120
RV_WIN        = 1     # QLIKE realized proxy：當期 z²（= 1 表示不滾動）



d = 1.0 / 10

KAPPA_SCALE = 1.0   # κ 初始中心縮小
SIGMA_SCALE = 1.0   # σ 初始中心放大
MIN_VAR     = 1e-4  # v_s 下界（對應最低日波動度 √(1e-6) ≈ 0.1%）

def _est_sv_params(y_s, v_s):
    mu_e    = float(np.mean(y_s))

    # Step 0: 去均值 + EWMA 代理變異數（λ=0.94）
    r       = y_s - mu_e
    proxy   = r**2
    lam     = 0.94
    rv      = pd.Series(proxy).ewm(alpha=1 - lam).mean().values
    rv      = np.clip(rv, MIN_VAR, None)

    # Step 2: θ — 未平滑 proxy 均值（最無偏）
    theta_e = float(max(np.mean(proxy), MIN_VAR))

    # Step 3: κ — AR(1) 斜率回推（斜率 = 1−κ）
    x_ar    = rv[:-1]
    yv      = rv[1:]
    b       = np.cov(x_ar, yv)[0, 1] / np.var(x_ar)
    kappa_e = float(np.clip(1.0 - b, 1e-3, 0.999))

    # Step 4: σ — AR(1) 殘差對 √V_{t-1} 做尺度匹配
    ar1_int = float(np.mean(yv) - b * np.mean(x_ar))
    u       = yv - ar1_int - b * x_ar
    sigma_e = float(np.sqrt(np.mean(u**2 / np.clip(x_ar, MIN_VAR, None))))
    sigma_e = max(sigma_e, 1e-4)

    # Step 5: ρ — 標準化報酬新息與 V 新息的相關
    eps1     = r[1:] / np.sqrt(np.clip(x_ar, MIN_VAR, None))
    eta      = u / (sigma_e * np.sqrt(np.clip(x_ar, MIN_VAR, None)))
    corr_lev = np.corrcoef(eps1, eta)[0, 1]
    rho_e    = float(np.clip(corr_lev if np.isfinite(corr_lev) else 0.0, -0.999, 0.999))

    kappa_e *= KAPPA_SCALE
    sigma_e  = max(sigma_e * SIGMA_SCALE, 1e-4)

    # Step 6: 軟性 Feller — 2κθ ≥ σ²；不滿足則縮 σ
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

# # ─── 固定初始值（來自 MSAEtimebreak.py）───
# C1_FIXED = np.array([ 0.001,  0.05,   0.0005, 0.0002, -0.5 ])   # theta1 固定中心
# C2_FIXED = np.array([-0.001,  0.01,   0.001,  0.0005, -0.25])   # theta2 固定中心
# X0_1 = 0.0005                                                     # asset1 初始波動度中心
# X0_2 = 0.001                                                      # asset2 初始波動度中心
# v1_arr = np.array([0.001*d, 0.05*d, 0.0005*d, 0.0002*d,  0.5*d])
# v2_arr = np.array([0.001*d, 0.01*d,  0.001*d, 0.0005*d, 0.25*d])
# v_all1 = v1_arr**2 * d
# v_all2 = v2_arr**2 * d



R          = np.zeros((Y, n_cop))

logLik     = np.zeros((Y, n_cop))
logLik_all = np.zeros(Y)
# logLik_m1  = np.zeros((Y, n_cop))   # Sklar 分解：資產1 邊際
# logLik_m2  = np.zeros((Y, n_cop))   # Sklar 分解：資產2 邊際
# logLik_cop = np.zeros((Y, n_cop))   # Sklar 分解：copula 貢獻

iter_arr   = np.zeros((Y, n_cop), dtype=int)

theta1_estimate = np.zeros((Y, k, n_cop))
theta2_estimate = np.zeros((Y, k, n_cop))

ks_stat  = np.zeros((Y, 2))
ks_pval  = np.zeros((Y, 2))
lb_pval  = np.zeros((Y, 2))
lb2_pval = np.zeros((Y, 2))




# ──── 資料迴圈(最外圍)  ─────────────────────────────────────────────────────────────────────────────────




for j in range(Y):

    #########################   ( 0.2 ) 迴圈中變數   ##################################################

    y = np.column_stack([y1[:, j], y2[:, j]])

    v1j = max(float(var1_emp[j]), MIN_VAR)
    v2j = max(float(var2_emp[j]), MIN_VAR)

    c1, v1_arr = _est_sv_params(y1[:, j], v1j)
    c2, v2_arr = _est_sv_params(y2[:, j], v2j)
    v_all1 = v1_arr**2 * d
    v_all2 = v2_arr**2 * d

    x      = np.zeros((N, T, p))
    mu     = np.zeros((N, T, p))

    theta1 = np.zeros((N, T, k))
    theta2 = np.zeros((N, T, k))
    m1     = np.zeros((N, T, k))
    m2     = np.zeros((N, T, k))

    rho_Ms = np.zeros((N, n_cop))


    # 抽樣兩個市場的初始波動度、參數  ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆ ☆


    #    波動度    #
    x[0, :, 0] = np.random.normal(v1j, v1j*d, T)
    neg = x[0, :, 0] < 0
    x[0, neg, 0] = 2*v1j - x[0, neg, 0]

    x[0, :, 1] = np.random.normal(v2j, v2j*d, T)
    neg = x[0, :, 1] < 0
    x[0, neg, 1] = 2*v2j - x[0, neg, 1]
    # x[0, :, 0] = np.random.normal(X0_1, X0_1*d, T)
    # neg = x[0, :, 0] < 0
    # x[0, neg, 0] = 2*X0_1 - x[0, neg, 0]
    # x[0, :, 1] = np.random.normal(X0_2, X0_2*d, T)
    # neg = x[0, :, 1] < 0
    # x[0, neg, 1] = 2*X0_2 - x[0, neg, 1]

    mu[0] = x[0]


    #    參數    #
    theta1[0, :, 0] = np.random.normal(c1[0], v1_arr[0], T)
    theta1[0, :, 1] = np.clip(np.random.gamma(c1[1]**2/v1_arr[1]**2, v1_arr[1]**2/c1[1], T), 1e-6, np.inf)
    theta1[0, :, 2] = np.clip(np.random.gamma(c1[2]**2/v1_arr[2]**2, v1_arr[2]**2/c1[2], T), 1e-6, np.inf)
    theta1[0, :, 3] = np.clip(np.random.gamma(c1[3]**2/v1_arr[3]**2, v1_arr[3]**2/c1[3], T), 1e-6, np.inf)
    _a, _b = c1[4] - np.sqrt(3)*v1_arr[4], c1[4] + np.sqrt(3)*v1_arr[4]
    theta1[0, :, 4] = np.clip(np.random.uniform(_a, _b, T), -0.999, 0.999)

    theta2[0, :, 0] = np.random.normal(c2[0], v2_arr[0], T)
    theta2[0, :, 1] = np.clip(np.random.gamma(c2[1]**2/v2_arr[1]**2, v2_arr[1]**2/c2[1], T), 1e-6, np.inf)
    theta2[0, :, 2] = np.clip(np.random.gamma(c2[2]**2/v2_arr[2]**2, v2_arr[2]**2/c2[2], T), 1e-6, np.inf)
    theta2[0, :, 3] = np.clip(np.random.gamma(c2[3]**2/v2_arr[3]**2, v2_arr[3]**2/c2[3], T), 1e-6, np.inf)
    _a, _b = c2[4] - np.sqrt(3)*v2_arr[4], c2[4] + np.sqrt(3)*v2_arr[4]
    theta2[0, :, 4] = np.clip(np.random.uniform(_a, _b, T), -0.999, 0.999)
    # # theta1[0, :, 0] = np.random.normal(C1_FIXED[0], v1_arr[0], T)
    # # theta1[0, :, 1] = np.clip(np.random.gamma(C1_FIXED[1]**2/v1_arr[1]**2, v1_arr[1]**2/C1_FIXED[1], T), 1e-6, np.inf)
    # # theta1[0, :, 2] = np.clip(np.random.gamma(C1_FIXED[2]**2/v1_arr[2]**2, v1_arr[2]**2/C1_FIXED[2], T), 1e-6, np.inf)
    # # theta1[0, :, 3] = np.clip(np.random.gamma(C1_FIXED[3]**2/v1_arr[3]**2, v1_arr[3]**2/C1_FIXED[3], T), 1e-6, np.inf)
    # # _a, _b = C1_FIXED[4] - np.sqrt(3)*v1_arr[4], C1_FIXED[4] + np.sqrt(3)*v1_arr[4]
    # # theta1[0, :, 4] = np.clip(np.random.uniform(_a, _b, T), -0.999, 0.999)
    # # theta2[0, :, 0] = np.random.normal(C2_FIXED[0], v2_arr[0], T)
    # # theta2[0, :, 1] = np.clip(np.random.gamma(C2_FIXED[1]**2/v2_arr[1]**2, v2_arr[1]**2/C2_FIXED[1], T), 1e-6, np.inf)
    # # theta2[0, :, 2] = np.clip(np.random.gamma(C2_FIXED[2]**2/v2_arr[2]**2, v2_arr[2]**2/C2_FIXED[2], T), 1e-6, np.inf)
    # # theta2[0, :, 3] = np.clip(np.random.gamma(C2_FIXED[3]**2/v2_arr[3]**2, v2_arr[3]**2/C2_FIXED[3], T), 1e-6, np.inf)
    # # _a, _b = C2_FIXED[4] - np.sqrt(3)*v2_arr[4], C2_FIXED[4] + np.sqrt(3)*v2_arr[4]
    # # theta2[0, :, 4] = np.clip(np.random.uniform(_a, _b, T), -0.999, 0.999)


    #########################  ( 0.3 ) —  隨機變數  ############################################


    r2 = np.random.randn(N, T, p)

    x0_init      = x[0].copy()
    theta1_0init = theta1[0].copy()
    theta2_0init = theta2[0].copy()

    best_ci_so_far = 0
    x_best       = None
    theta1_best  = None
    theta2_best  = None
    theta1_final = None   # (N, k) 所有粒子末期參數（最佳 copula）
    theta2_final = None
    x_final      = None   # (N, p) 所有粒子末期狀態




    # ── Copula 迴圈 ─────────────────────────────────────────────────────────────────────────────────




    for ci, c in enumerate(COPULA_NAMES):

        # 每個 copula 從相同初始狀態出發
        x[0]      = x0_init.copy()
        theta1[0] = theta1_0init.copy()
        theta2[0] = theta2_0init.copy()
        mu[0]     = x0_init.copy()


        ###################### ( 0.4 ) — 初始權重 ####################################################

        w       = np.zeros((N, T))
        w[0, :] = 1.0 / T


        # ── 粒子迴圈 ────────────────────────────────────────────────────────────────────────────────


        for i in range(1, N):


            ################## ( 1.1 ) — 平滑參數（ resampling 前 ） ##################################

            theta_mean = w[i-1] @ theta1[i-1]
            m1[i-1] = a * theta1[i-1] + (1-a) * theta_mean

            theta_mean = w[i-1] @ theta2[i-1]
            m2[i-1] = a * theta2[i-1] + (1-a) * theta_mean


            ################## ( 1.2 ) — 預測 mu（ 下一期波動度 ） #####################################

            mu[i, :, 0] = x[i-1, :, 0] + m1[i-1, :, 1] * (m1[i-1, :, 2] - x[i-1, :, 0])
            mu[i, :, 1] = x[i-1, :, 1] + m2[i-1, :, 1] * (m2[i-1, :, 2] - x[i-1, :, 1])


            ################## ( 1.3 ) — copula likelihood g  #####################################

            SIGMA_g = np.sqrt(np.maximum(mu[i], 1e-10))
            MU_g    = np.column_stack([m1[i-1, :, 0]
                                     ,m2[i-1, :, 0]]) - mu[i] / 2

            u_t     = np.clip(stats.norm.cdf(y, MU_g, SIGMA_g), 1e-4, 1-1e-4)

            y_p     = stats.norm.pdf(y, MU_g, SIGMA_g)

            cop_pdf, _ = copulafitall3(COPULA_NAMES, c, u_t)

            g     = w[i-1] * cop_pdf * y_p[:, 0] * y_p[:, 1]
            g_sum = g.sum()
            g     = g / g_sum if g_sum > 0 else np.ones(T) / T


            ################ ( 2.1 ) — 重抽樣參數、生成波動度 ##################################

            rs = np.random.choice(T, T, replace=True, p=g)
            theta1[i-1] = theta1[i-1, rs]
            theta2[i-1] = theta2[i-1, rs]

            x[i-1, 0] = mu[i, 0]

            # 生成波動度
            for t in range(1, T):

                sx1 = np.sqrt(max(x[i-1, t-1, 0], 1e-8))

                r1a = (y[t, 0] - (theta1[i-1, t, 0] - 0.5*x[i-1, t-1, 0])) / sx1

                x[i-1, t, 0] = (x[i-1, t-1, 0]
                    + theta1[i-1, t, 1] * (theta1[i-1, t, 2] - x[i-1, t-1, 0])
                    + theta1[i-1, t, 3] * sx1 * (theta1[i-1, t, 4] * r1a
                    + np.sqrt(max(1 - theta1[i-1, t, 4]**2, 0)) * r2[i-1, t, 0]))


                sx2 = np.sqrt(max(x[i-1, t-1, 1], 1e-8))

                r1b = (y[t, 1] - (theta2[i-1, t, 0] - 0.5*x[i-1, t-1, 1])) / sx2

                x[i-1, t, 1] = (x[i-1, t-1, 1]
                    + theta2[i-1, t, 1] * (theta2[i-1, t, 2] - x[i-1, t-1, 1])
                    + theta2[i-1, t, 3] * sx2 * (theta2[i-1, t, 4] * r1b
                    + np.sqrt(max(1 - theta2[i-1, t, 4]**2, 0)) * r2[i-1, t, 1]))


            ################## ( 2.2 ) — 計算平滑參數、波動度（resampling 後） #############################

            theta_mean = np.mean(theta1[i-1], axis=0)
            m1[i-1]    = a * theta1[i-1] + (1-a) * theta_mean
            v_1        = np.maximum(np.var(theta1[i-1], axis=0, ddof=1), v_all1)

            theta_mean = np.mean(theta2[i-1], axis=0)
            m2[i-1]    = a * theta2[i-1] + (1-a) * theta_mean
            v_2        = np.maximum(np.var(theta2[i-1], axis=0, ddof=1), v_all2)

            mu[i, :, 0] = x[i-1, :, 0] + m1[i-1, :, 1] * (m1[i-1, :, 2] - x[i-1, :, 0])
            mu[i, :, 1] = x[i-1, :, 1] + m2[i-1, :, 1] * (m2[i-1, :, 2] - x[i-1, :, 1])

            negative_index2 = mu[i] < 0
            mu[i] = np.abs(mu[i])


            ################## ( 2.3 ) — 抽樣下一期參數 ##################################### ☆ ☆ ☆

            sd1 = np.sqrt(h**2 * v_1)
            sd2 = np.sqrt(h**2 * v_2)


            theta1[i, :, 0] = np.random.normal(m1[i-1, :, 0], sd1[0])
            theta2[i, :, 0] = np.random.normal(m2[i-1, :, 0], sd2[0])


            theta1[i, :, 1] = np.clip(np.random.gamma(m1[i-1,:,1]**2/sd1[1]**2, sd1[1]**2/m1[i-1,:,1]), 1e-6, np.inf)
            theta2[i, :, 1] = np.clip(np.random.gamma(m2[i-1,:,1]**2/sd2[1]**2, sd2[1]**2/m2[i-1,:,1]), 1e-6, np.inf)


            theta1[i, :, 2] = np.clip(np.random.gamma(m1[i-1,:,2]**2/sd1[2]**2, sd1[2]**2/m1[i-1,:,2]), 1e-6, np.inf)
            theta2[i, :, 2] = np.clip(np.random.gamma(m2[i-1,:,2]**2/sd2[2]**2, sd2[2]**2/m2[i-1,:,2]), 1e-6, np.inf)


            theta1[i, :, 3] = np.clip(np.random.gamma(m1[i-1,:,3]**2/sd1[3]**2, sd1[3]**2/m1[i-1,:,3]), 1e-6, np.inf)
            theta2[i, :, 3] = np.clip(np.random.gamma(m2[i-1,:,3]**2/sd2[3]**2, sd2[3]**2/m2[i-1,:,3]), 1e-6, np.inf)


            _a1 = m1[i-1, :, 4] - np.sqrt(3)*sd1[4];  _b1 = m1[i-1, :, 4] + np.sqrt(3)*sd1[4]
            theta1[i, :, 4] = np.clip(np.random.uniform(_a1, _b1), -0.999, 0.999)
            _a2 = m2[i-1, :, 4] - np.sqrt(3)*sd2[4];  _b2 = m2[i-1, :, 4] + np.sqrt(3)*sd2[4]
            theta2[i, :, 4] = np.clip(np.random.uniform(_a2, _b2), -0.999, 0.999)


            ################## ( 2.4 ) — 用新參數生成波動度 ###########################################

            x[i, 0] = mu[i, 0]

            for t in range(1, T):

                sx1 = np.sqrt(max(x[i, t-1, 0], 1e-8))

                r1a = (y[t, 0] - (theta1[i, t, 0] - 0.5*x[i, t-1, 0])) / sx1

                x[i, t, 0] = (x[i, t-1, 0]
                    + theta1[i, t, 1] * (theta1[i, t, 2] - x[i, t-1, 0])
                    + theta1[i, t, 3] * sx1 * (theta1[i, t, 4] * r1a
                    + np.sqrt(max(1 - theta1[i, t, 4]**2, 0)) * r2[i, t, 0]))


                sx2 = np.sqrt(max(x[i, t-1, 1], 1e-8))

                r1b = (y[t, 1] - (theta2[i, t, 0] - 0.5*x[i, t-1, 1])) / sx2

                x[i, t, 1] = (x[i, t-1, 1]
                    + theta2[i, t, 1] * (theta2[i, t, 2] - x[i, t-1, 1])
                    + theta2[i, t, 3] * sx2 * (theta2[i, t, 4] * r1b
                    + np.sqrt(max(1 - theta2[i, t, 4]**2, 0)) * r2[i, t, 1]))

            negative_index1 = x[i] < 0
            x[i] = np.abs(x[i])


            ################## ( 3.1 ) — 計算 copula ( 用來計算權重 ) ##############################

            MU1  = np.column_stack([theta1[i, :, 0], theta2[i, :, 0]]) - x[i] / 2
            SIG1 = np.sqrt(np.maximum(x[i], 1e-10))
            u_t1 = np.clip(stats.norm.cdf(y, MU1, SIG1), 1e-4, 1-1e-4)
            yp1  = stats.norm.pdf(y, MU1, SIG1)

            cop1, rho_val = copulafitall3(COPULA_NAMES, c, u_t1)
            c1 = cop1 * yp1[:, 0] * yp1[:, 1]
            rho_Ms[i, ci] = rho_val

            MU2  = np.column_stack([m1[i-1, :, 0], m2[i-1, :, 0]]) - mu[i] / 2
            SIG2 = np.sqrt(np.maximum(mu[i], 1e-10))
            u_t2 = np.clip(stats.norm.cdf(y, MU2, SIG2), 1e-4, 1-1e-4)
            yp2  = stats.norm.pdf(y, MU2, SIG2)

            cop2, _ = copulafitall3(COPULA_NAMES, c, u_t2)
            c2 = cop2 * yp2[:, 0] * yp2[:, 1]

            w[i] = np.where(c2 > 0, c1 / c2, 0.0)
            w[i, ~np.isfinite(w[i])]    = 0.0
            w[i, negative_index1[:, 0]] = 0.0
            w[i, negative_index1[:, 1]] = 0.0
            w[i, negative_index2[:, 0]] = 0.0
            w[i, negative_index2[:, 1]] = 0.0
            w_sum = w[i].sum()
            w[i] = w[i] / w_sum if w_sum > 0 else np.ones(T) / T


        # ── logLik (still in copula loop) ──────────────────────────────────────────────────────────


        iter_arr[j, ci] = N - 1
        conv_idx = N - 1
        xN = x[conv_idx]

        MU_l  = np.column_stack([theta1[conv_idx, :, 0], theta2[conv_idx, :, 0]]) - xN / 2
        SIG_l = np.sqrt(np.maximum(xN, 1e-10))
        u_tl  = np.clip(stats.norm.cdf(y, MU_l, SIG_l), 1e-4, 1-1e-4)
        yp_l  = stats.norm.pdf(y, MU_l, SIG_l)
        cop_l, _ = copulafitall3(COPULA_NAMES, c, u_tl)
        joint_pdf = np.maximum(cop_l * yp_l[:, 0] * yp_l[:, 1], np.finfo(float).tiny)
        logLik[j, ci]    = np.sum(np.log(joint_pdf))



        # 更改--不用最後一筆(Gumbel)作為估計值
        if x_best is None or logLik[j, ci] > logLik[j, best_ci_so_far]:
            best_ci_so_far = ci
            x_best       = x[conv_idx].copy()
            theta1_best  = theta1[conv_idx].copy()
            theta2_best  = theta2[conv_idx].copy()
            theta1_final = theta1[:, -1, :].copy()   # (N, k) 所有粒子 t=T-1 的參數後驗
            theta2_final = theta2[:, -1, :].copy()
            x_final      = x[:, -1, :].copy()        # (N, p) 所有粒子 t=T-1 的狀態後驗


        R[j, ci] = rho_Ms[conv_idx, ci]
        theta1_estimate[j, :, ci] = np.mean(theta1[conv_idx], axis=0)
        theta2_estimate[j, :, ci] = np.mean(theta2[conv_idx], axis=0)




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
for ci, name in enumerate(COPULA_NAMES):
    print(f'  {name}: τ = {R[0, ci]:.4f}')

# ── logLik 與最佳 Copula ──────────────────────────────────────────────────────
print('\n── 各 Copula logLik ──')
for ci, name in enumerate(COPULA_NAMES):
    print(f'  {name}: {logLik[0, ci]:.4f}')

best_ci_out = int(np.argmax(logLik[0]))
print(f'\n最佳 Copula: {COPULA_NAMES[best_ci_out]}  (logLik = {np.max(logLik[0]):.4f})')

# ── 殘差診斷（最佳 Copula）────────────────────────────────────────────────────
print(f'\n── 殘差診斷（最佳 Copula: {COPULA_NAMES[best_ci_out]}）──')
asset_labels = ['資產1', '資產2']
for a, aname in enumerate(asset_labels):
    print(f'\n  {aname}:')
    print(f'    KS  (PIT vs U[0,1]):  stat={ks_stat[0, a]:.4f}  p={ks_pval[0, a]:.4f}{"  （拒絕）" if ks_pval[0, a] < 0.05 else "  （未拒絕）"}')
    print(f'    LB(20)  殘差自相關:    p={lb_pval[0, a]:.4f}{"  （拒絕）" if lb_pval[0, a] < 0.05 else "  （未拒絕）"}')
    print(f'    LB²(20) ARCH 效果:     p={lb2_pval[0, a]:.4f}{"  （拒絕）" if lb2_pval[0, a] < 0.05 else "  （未拒絕）"}')

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
        -(mu_hat[:, a] + np.mean(z[z[:, a] <= z_alpha[a], a]) * sig_hat[:, a])
        for a in range(2)
    ])                                                               # (T, 2)
    pct = int((1 - alpha) * 100)
    line_var = f'    VaR {pct}%:  {cond_VaR[:, 0].mean():10.6f}  {cond_VaR[:, 1].mean():10.6f}'
    line_es  = f'    ES  {pct}%:  {cond_ES[:, 0].mean():10.6f}  {cond_ES[:, 1].mean():10.6f}'
    print(line_var)
    print(line_es)


# ── B. VaR 回測：Kupiec POF & Christoffersen CC 檢定 ──────────────────────────
def _kupiec_pof(hit, alpha):
    """Kupiec (1995) POF：無條件涵蓋，LR_uc ~ χ²(1)。"""
    T_, x = len(hit), int(hit.sum())
    p = x / T_
    if x == 0 or x == T_:
        return np.nan, np.nan
    lr = -2.0 * (x * np.log(alpha / p) + (T_ - x) * np.log((1 - alpha) / (1 - p)))
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
        w_lag   = 1.0 - lag / h_horizon
        gamma_l = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        lrv    += 2.0 * w_lag * gamma_l
    lrv    = max(lrv, 1e-20)
    dm     = d_bar / np.sqrt(lrv / n)
    p_two  = 2.0 * float(stats.t.sf(abs(dm), df=n - 1))
    p_less = float(stats.t.cdf(dm, df=n - 1))   # H₁: loss1 < loss2
    return float(dm), p_two, p_less


def _dw(s):
    """字串顯示寬度（CJK 全形字元算 2，其餘算 1）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in str(s))

def _lj(s, w): return str(s) + ' ' * max(0, w - _dw(str(s)))
def _rj(s, w): return ' ' * max(0, w - _dw(str(s))) + str(s)
def _fl(v):    return f'{v:.4f}' if not np.isnan(v) else 'nan'
def _sig(p):   return '*' if (not np.isnan(p) and p < 0.05) else ' '

# 各欄顯示寬度（display width）
#        資產  違反/T  違反率  LR_POF  p_POF  LR_ind  p_ind  LR_cc  p_cc
WCOL = [  14,     9,     7,      8,     7,      8,     7,     8,    7]
HCOL = ['資產', '違反/T', '違反率',
        'LR_POF', 'p_POF', 'LR_ind', 'p_ind', 'LR_cc', 'p_cc']

for alpha in VaR_LEVELS:
    pct      = int((1 - alpha) * 100)
    z_alpha  = np.quantile(z, alpha, axis=0)
    cond_VaR = -(mu_hat + z_alpha * sig_hat)

    series = [
        ('資產1', y[:, 0], cond_VaR[:, 0]),
        ('資產2', y[:, 1], cond_VaR[:, 1]),
    ]
    SEP = '  '
    print(f'\n── VaR 回測 {pct}%（α={alpha:.2f}）—— Kupiec POF & Christoffersen CC ──')
    print('    (* p<0.05 拒絕 H₀)')
    print('    ' + SEP.join(_rj(h, w) for h, w in zip(HCOL, WCOL)))
    print('    ' + SEP.join('-' * w for w in WCOL))

    for lab, ret, var_ in series:
        hit           = (ret < -var_)
        x             = int(hit.sum())
        p_hat         = x / T
        lr_uc, p_uc   = _kupiec_pof(hit, alpha)
        lr_ind, p_ind = _christoffersen_ind(hit)
        lr_cc = (lr_uc + lr_ind) if not (np.isnan(lr_uc) or np.isnan(lr_ind)) else np.nan
        p_cc  = float(1.0 - stats.chi2.cdf(lr_cc, df=2)) if not np.isnan(lr_cc) else np.nan

        row = [
            _lj(lab,                           WCOL[0]),
            _rj(f'{x}/{T}',                    WCOL[1]),
            _rj(f'{p_hat:.4f}',                WCOL[2]),
            _rj(_fl(lr_uc),                    WCOL[3]),
            _rj(_fl(p_uc)  + _sig(p_uc),       WCOL[4]),
            _rj(_fl(lr_ind),                   WCOL[5]),
            _rj(_fl(p_ind) + _sig(p_ind),      WCOL[6]),
            _rj(_fl(lr_cc),                    WCOL[7]),
            _rj(_fl(p_cc)  + _sig(p_cc),       WCOL[8]),
        ]
        print('    ' + SEP.join(row))


# ─────────────────────────────────────────────────────────────────────────────
# ── SV 參數有效性檢定（κ > 0, θ > 0, σ > 0）────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#
# def _ttest_gt0(arr):
#     """單側 t 檢定：H₀: μ ≤ 0，H₁: μ > 0。回傳 (均值, t, p)。"""
#     t, p = stats.ttest_1samp(arr, popmean=0, alternative='greater')
#     return float(np.mean(arr)), float(t), float(p)
#
# #        κ          θ          σ
# PIDX = [1,         2,         3        ]
# PLAB = ['κ > 0', 'θ > 0', 'σ > 0']
# PDSC = ['均值回歸為正', '長期變異數為正', '波動率為正']
#
# #      label       估計值  t-stat  p值
# PW   = [16,        10,     8,      7]
#
# print(f'\n── SV 參數有效性檢定（最佳 Copula: {COPULA_NAMES[best_ci_out]}）──')
# print('    H₀: 各參數 ≤ 0；單側 t 檢定，* p < 0.05 拒絕 H₀（確認條件成立）')
# print(f'\n    {"":16s}  {"─── 資產1 ───":^27s}  {"─── 資產2 ───":^27s}')
# print('    ' + '  '.join([
#     _rj('檢定', PW[0]), _rj('估計值', PW[1]), _rj('t-stat', PW[2]), _rj('p 值', PW[3]),
#     _rj('估計值', PW[1]), _rj('t-stat', PW[2]), _rj('p 值', PW[3]),
# ]))
# print('    ' + '  '.join('-' * w for w in [PW[0]] + PW[1:] * 2))
#
# for pidx, plab, pdsc in zip(PIDX, PLAB, PDSC):
#     m1, t1, p1 = _ttest_gt0(theta1_best[:, pidx])
#     m2, t2, p2 = _ttest_gt0(theta2_best[:, pidx])
#     label = f'{plab}（{pdsc}）'
#     row = [
#         _lj(label,          PW[0]),
#         _rj(f'{m1:.6f}',    PW[1]),
#         _rj(f'{t1:.4f}',    PW[2]),
#         _rj(f'{p1:.4f}' + _sig(p1), PW[3]),
#         _rj(f'{m2:.6f}',    PW[1]),
#         _rj(f'{t2:.4f}',    PW[2]),
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


# ─────────────────────────────────────────────────────────────────────────────
# ── Volatility Forecasting & QLIKE ───────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

PRED_PATH = r"C:\Users\user\PycharmProjects\sd-copf\Empirical\data\real_data2-1_pred.xlsx"
y_pred1 = pd.read_excel(PRED_PATH, sheet_name='return1', header=None).values.ravel()
y_pred2 = pd.read_excel(PRED_PATH, sheet_name='return2', header=None).values.ravel()
y_pred  = np.column_stack([y_pred1, y_pred2])
T_pred  = len(y_pred1)

kappa1_hat = float(np.mean(theta1_best[:, 1]))
long1_hat  = float(np.mean(theta1_best[:, 2]))
kappa2_hat = float(np.mean(theta2_best[:, 1]))
long2_hat  = float(np.mean(theta2_best[:, 2]))
kappa1_c   = float(np.clip(kappa1_hat, 1e-8, 10.0))
kappa2_c   = float(np.clip(kappa2_hat, 1e-8, 10.0))

# ── A. 樣本內 1-step-ahead forecast（用於 IS QLIKE 對照）─────────────────────
h_is = np.empty((T, 2))
h_is[0] = x_best[0]
for t in range(1, T):
    h_is[t, 0] = x_best[t-1, 0] + kappa1_c * (long1_hat - x_best[t-1, 0])
    h_is[t, 1] = x_best[t-1, 1] + kappa2_c * (long2_hat - x_best[t-1, 1])
h_is = np.maximum(h_is, 1e-10)

# ── QLIKE 代理：去漂移殘差 z² 滾動均值（與模型 Var[R|F_{t-1}]=V_{t-1} 無偏一致）
# z_{t,p} = y[t,p] − (μ_{t-1,p} − V_{t-1,p}/2)，E[z²|F_{t-1}] = V_{t-1,p}
mu1_is   = float(np.mean(theta1_best[:, 0]))   # 常數後驗均值，與 OOS 對齊
mu2_is   = float(np.mean(theta2_best[:, 0]))
drift_is = np.column_stack([
    mu1_is - 0.5 * x_best[:-1, 0],                # (T-1,) 資產1 條件均值
    mu2_is - 0.5 * x_best[:-1, 1]                 # (T-1,) 資產2 條件均值
])
z_is    = y[1:] - drift_is                          # (T-1, 2) 去漂移殘差
rv_is_m = np.column_stack([
    pd.Series(z_is[:, 0]**2).rolling(RV_WIN).mean().values,
    pd.Series(z_is[:, 1]**2).rolling(RV_WIN).mean().values
])   # (T-1, 2)，對齊 h_is[1:]；前 RV_WIN-1 列為 NaN

qlike_is   = np.nanmean(np.log(h_is[1:]) + rv_is_m / h_is[1:], axis=0)
h_const_is = np.nanmean(rv_is_m, axis=0)

# ── B. 滾動 OOS 序列粒子濾波（固定參數，逐期以觀測值更新波動度狀態）────────
# h_roll[t] = E[h_{T+t+1} | h_{T+t}]（N 粒子 CIR 期望值均值）
# 觀測到 y_pred[t] 後，以含槓桿效應的 SV 方程更新粒子狀態
r2_oos       = np.random.randn(T_pred, N, 2)
x_oos        = x_final.copy()            # (N, 2) IS 末期後驗狀態
x_oos_states = np.empty((T_pred, N, 2)) # 各 OOS 期濾波狀態（觀測前）
h_roll       = np.empty((T_pred, 2))

for t in range(T_pred):
    x_oos_states[t] = x_oos   # 儲存觀測 y_pred[t] 前的狀態

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

# OOS 實現代理：去漂移殘差 z² 滾動均值
# 條件均值 = N 粒子後驗 μ̂ − V_{T+t}/2（V_{T+t} 為 x_oos_states[t] 後驗均值）
mu1_oos   = float(np.mean(theta1_final[:, 0]))
mu2_oos   = float(np.mean(theta2_final[:, 0]))
drift_oos = np.column_stack([
    mu1_oos - 0.5 * np.mean(x_oos_states[:, :, 0], axis=1),   # (T_pred,)
    mu2_oos - 0.5 * np.mean(x_oos_states[:, :, 1], axis=1)
])
z_pred    = y_pred - drift_oos                                  # (T_pred, 2) 去漂移殘差
rv_pred_m = np.column_stack([
    pd.Series(z_pred[:, 0]**2).rolling(RV_WIN).mean().values,
    pd.Series(z_pred[:, 1]**2).rolling(RV_WIN).mean().values
])   # (T_pred, 2)，前 RV_WIN-1 列為 NaN

# ── C. 樣本內 vs 樣本外 QLIKE 比較表 ─────────────────────────────────────────
qlike_oos       = np.nanmean(np.log(h_roll)    + rv_pred_m / h_roll,    axis=0)
qlike_bench_is  = np.log(h_const_is) + np.nanmean(rv_is_m  / h_const_is, axis=0)
qlike_bench_oos = np.log(h_const_is) + np.nanmean(rv_pred_m    / h_const_is, axis=0)

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
loss_sv_is     = np.log(h_is[1:])   + rv_is_m   / h_is[1:]    # (T-1, 2)
loss_bench_is  = np.log(h_const_is) + rv_is_m   / h_const_is  # (T-1, 2)
loss_sv_oos    = np.log(h_roll)     + rv_pred_m  / h_roll      # (T_pred, 2)
loss_bench_oos = np.log(h_const_is) + rv_pred_m  / h_const_is # (T_pred, 2)

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
# 從 OOS 第 t 期後驗狀態出發，CIR 解析 h 步期望值：
#   E[h_{T+t+h} | h_{T+t}, θ] = θ + (h_{T+t} − θ)(1−κ)^h
# realized proxy：rv_pred_m[t+h-1]（0-based），對 N 粒子取均值後計算 QLIKE

print('\n── Rolling h-step Forecast QLIKE & DM 檢定（樣本外，CIR 解析解）──')
print(f'  {"h":>4s}  {"Q1":>12s}  {"DM1":>9s}  {"p2_1":>7s}  {"pl_1":>7s}'
      f'  {"Q2":>12s}  {"DM2":>9s}  {"p2_2":>7s}  {"pl_2":>7s}  {"n":>6s}')
print('  ' + '-' * 100)

for h in FCST_HORIZONS:
    n_periods = T_pred - h + 1
    if n_periods <= 0:
        continue

    st  = x_oos_states[:n_periods]          # (n_periods, N, 2)
    k1  = np.clip(theta1_final[None, :, 1], 0.0, 1.0)   # (1, N)
    th1 = theta1_final[None, :, 2]                       # (1, N)
    k2  = np.clip(theta2_final[None, :, 1], 0.0, 1.0)
    th2 = theta2_final[None, :, 2]

    fcast1 = th1 + (st[:, :, 0] - th1) * (1 - k1)**h   # (n_periods, N)
    fcast2 = th2 + (st[:, :, 1] - th2) * (1 - k2)**h

    fh1 = np.mean(np.maximum(fcast1, 1e-10), axis=1)    # (n_periods,)
    fh2 = np.mean(np.maximum(fcast2, 1e-10), axis=1)

    rv_tgt     = rv_pred_m[h-1:h-1+n_periods]           # (n_periods, 2)
    loss_sv_h1 = np.log(fh1) + rv_tgt[:, 0] / fh1
    loss_sv_h2 = np.log(fh2) + rv_tgt[:, 1] / fh2
    loss_bch_h1= np.log(h_const_is[0]) + rv_tgt[:, 0] / h_const_is[0]
    loss_bch_h2= np.log(h_const_is[1]) + rv_tgt[:, 1] / h_const_is[1]

    q1 = float(np.nanmean(loss_sv_h1))
    q2 = float(np.nanmean(loss_sv_h2))
    dm_h1, p2_h1, pl_h1 = _dm_test(loss_sv_h1, loss_bch_h1, h_horizon=h)
    dm_h2, p2_h2, pl_h2 = _dm_test(loss_sv_h2, loss_bch_h2, h_horizon=h)
    print(f'  {h:>4d}  {q1:12.6f}  {dm_h1:9.4f}  {p2_h1:7.4f}  {pl_h1:7.4f}{_sig(pl_h1)}'
          f'  {q2:12.6f}  {dm_h2:9.4f}  {p2_h2:7.4f}  {pl_h2:7.4f}{_sig(pl_h2)}  {n_periods:>6d}')

# ── Rolling 1-step QLIKE（樣本外，window W）──────────────────────────────────
WIN_OOS = min(WIN_OOS_MAX, T_pred // 4)
roll_qlike_oos = np.full((T_pred, 2), np.nan)
for t in range(WIN_OOS, T_pred):
    sl    = slice(t - WIN_OOS, t)
    h_sl  = h_roll[sl]
    rv_sl = rv_pred_m[sl]
    roll_qlike_oos[t, 0] = float(np.nanmean(np.log(h_sl[:, 0]) + rv_sl[:, 0] / h_sl[:, 0]))
    roll_qlike_oos[t, 1] = float(np.nanmean(np.log(h_sl[:, 1]) + rv_sl[:, 1] / h_sl[:, 1]))

# ── 粒子 1-step 90% CI（向量化）───────────────────────────────────────────────
mu_all_1 = (x_oos_states[:, :, 0]
            + theta1_final[None, :, 1] * (theta1_final[None, :, 2] - x_oos_states[:, :, 0]))
mu_all_2 = (x_oos_states[:, :, 1]
            + theta2_final[None, :, 1] * (theta2_final[None, :, 2] - x_oos_states[:, :, 1]))
h_roll_q05 = np.column_stack([np.quantile(np.maximum(mu_all_1, 1e-10), 0.05, axis=1),
                               np.quantile(np.maximum(mu_all_2, 1e-10), 0.05, axis=1)])
h_roll_q95 = np.column_stack([np.quantile(np.maximum(mu_all_1, 1e-10), 0.95, axis=1),
                               np.quantile(np.maximum(mu_all_2, 1e-10), 0.95, axis=1)])

# ── IS 滾動 QLIKE（window WIN_OOS）───────────────────────────────────────────
T_is          = T - 1   # IS QLIKE 有效期數（t=1,...,T-1），對齊 h_is[1:] 與 rv_is_m
roll_qlike_is = np.full((T_is, 2), np.nan)
for t in range(WIN_OOS, T_is):
    sl = slice(t - WIN_OOS, t)
    h_sl_is  = h_is[1:][sl]
    rv_sl_is = rv_is_m[sl]
    roll_qlike_is[t, 0] = float(np.nanmean(np.log(h_sl_is[:, 0]) + rv_sl_is[:, 0] / h_sl_is[:, 0]))
    roll_qlike_is[t, 1] = float(np.nanmean(np.log(h_sl_is[:, 1]) + rv_sl_is[:, 1] / h_sl_is[:, 1]))

# ── 圖：IS 1-step forecast vs z² / rolling QLIKE ─────────────────────────────
t_is_arr = np.arange(1, T)   # t=1,...,T-1

fig_is, axes_is = plt.subplots(2, 2, figsize=(14, 8))
for ai, aname in enumerate(['資產1', '資產2']):
    _proxy_lbl_is = f'z² proxy (rolling {RV_WIN}d)' if RV_WIN > 1 else 'z² proxy (逐期，去漂移)'
    axes_is[0, ai].plot(t_is_arr, rv_is_m[:, ai],  label=_proxy_lbl_is,               alpha=0.6)
    axes_is[0, ai].plot(t_is_arr, h_is[1:, ai],    label='IS 1-step (best particle)',  linewidth=1.5)
    axes_is[0, ai].set_title(f'IS 1-step Forecast vs z² Proxy - {aname}')
    axes_is[0, ai].legend()

    axes_is[1, ai].plot(np.arange(T_is), roll_qlike_is[:, ai])
    axes_is[1, ai].set_title(f'IS Rolling QLIKE (W={WIN_OOS}) - {aname}')
    axes_is[1, ai].set_xlabel('t (IS)')

plt.suptitle('In-Sample Volatility Forecast & QLIKE', fontsize=13)
plt.tight_layout()
plt.show()

# ── 圖：OOS 滾動 1-step forecast vs realized RV / rolling QLIKE ──────────────
t_pred_arr = np.arange(T_pred)

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8))
for ai, aname in enumerate(['資產1', '資產2']):
    _proxy_lbl = f'z² proxy (rolling {RV_WIN}d)' if RV_WIN > 1 else 'z² proxy (逐期，去漂移)'
    axes3[0, ai].plot(t_pred_arr, rv_pred_m[:, ai],   label=_proxy_lbl,                          alpha=0.6)
    axes3[0, ai].plot(t_pred_arr, h_roll[:, ai],       label='Rolling 1-step (posterior mean)', linewidth=1.5)
    axes3[0, ai].fill_between(t_pred_arr,
                               h_roll_q05[:, ai], h_roll_q95[:, ai],
                               alpha=0.2, label='90% CI')
    axes3[0, ai].set_title(f'OOS 1-step Forecast vs z² Proxy - {aname}')
    axes3[0, ai].legend()

    axes3[1, ai].plot(t_pred_arr, roll_qlike_oos[:, ai])
    axes3[1, ai].set_title(f'OOS Rolling QLIKE (W={WIN_OOS}) - {aname}')
    axes3[1, ai].set_xlabel('t (OOS)')

plt.tight_layout()
plt.show()
