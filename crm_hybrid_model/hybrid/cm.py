"""Слой 1. Ёмкостная модель (CM) дебита жидкости без забойного давления.

Состояние скважины j — «потенциальный» дебит s_j (то, что скважина давала бы при работе
полный шаг). Баланс зоны дренирования:

    tau_j * ds_j/dt = w_j(t) - ue_j(t) * s_j,     w_j = sum_i f_ij * I_i(t) + e_j
    q_j(t) = u_j(t) * s_j(t)

    u_j   — доля времени работы за шаг (0 при остановке);
    ue_j  = u_j + kappa * (1 - u_j) — при остановке отбор не прекращается полностью:
            часть притока (kappa) уходит к соседям, поэтому давление растёт, но ограниченно;
    f_ij  — доля закачки нагнетательной i, приходящая к добывающей j (sum_j f_ij <= 1);
    e_j   — постоянный приток, не связанный с учтёнными нагнетательными (законтурная зона,
            закачка из-за границ участка);
    tau_j — постоянная времени.

При u = 1 это обычная CRMP (Сайарпур) без члена с забойным давлением — та же модель,
что в cost-БФА. Решение на шаге точное (экспоненциальное), шаг может быть любым.

Градиент функции потерь считается аналитически (сопряжённый проход назад по времени),
оптимизация — L-BFGS-B. Ограничения зашиты в параметризацию:
tau — сигмоида в [tau_min, tau_max], f — softmax по добывающим с «пустым» выходом (потери).
"""
import numpy as np
from scipy.optimize import minimize


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softplus(x):
    return np.logaddexp(0.0, x)


class CM:
    def __init__(self, mask, dist, step=7.0, tau_bounds=(3.0, 400.0), kappa_bounds=(0.05, 1.0),
                 l2_logit=1e-6):
        self.mask = mask                       # [I,P]
        self.dist = dist
        self.step = float(step)
        self.tau_lo, self.tau_hi = tau_bounds
        self.k_lo, self.k_hi = kappa_bounds
        self.l2 = l2_logit
        self.I, self.P = mask.shape
        self.nz = int(mask.sum())
        self.theta = None
        self.qs = None

    # ---------- параметры ----------
    def _unpack(self, th):
        P, nz = self.P, self.nz
        a, b, c = th[:P], th[P:2 * P], th[2 * P:3 * P]
        z = np.full((self.I, self.P), -np.inf)
        z[self.mask] = th[3 * P:3 * P + nz]
        k = th[3 * P + nz]
        return a, b, c, z, k

    def _params(self, th):
        a, b, c, z, k = self._unpack(th)
        sg = _sigmoid(a)
        tau = self.tau_lo + (self.tau_hi - self.tau_lo) * sg
        e = _softplus(b) * self.qs
        s0 = _softplus(c) * self.qs
        zmax = np.maximum(z.max(axis=1, keepdims=True), 0.0)
        ez = np.exp(z - zmax)
        f = ez / (np.exp(-zmax) + ez.sum(axis=1, keepdims=True))
        kap = self.k_lo + (self.k_hi - self.k_lo) * _sigmoid(k)
        return tau, e, s0, f, kap

    def set_params(self, tau, e, s0, f, kappa, qs):
        """Задать модель явно (для синтетических тестов): обратное преобразование к theta."""
        self.qs = np.asarray(qs, float)
        inv_sp = lambda x: np.log(np.expm1(np.maximum(x, 1e-8)))
        logit = lambda x: np.log(x / (1.0 - x))
        a = logit(np.clip((tau - self.tau_lo) / (self.tau_hi - self.tau_lo), 1e-6, 1 - 1e-6))
        rest = np.maximum(1.0 - f.sum(axis=1, keepdims=True), 1e-9)
        z = np.log(np.maximum(f, 1e-12) / rest)
        k = logit(np.clip((kappa - self.k_lo) / (self.k_hi - self.k_lo), 1e-6, 1 - 1e-6))
        self.theta = np.concatenate([a, inv_sp(e / self.qs), inv_sp(s0 / self.qs), z[self.mask], [k]])
        return self

    @property
    def tau(self):
        return self._params(self.theta)[0]

    @property
    def e(self):
        return self._params(self.theta)[1]

    @property
    def f(self):
        return self._params(self.theta)[3]

    @property
    def kappa(self):
        return self._params(self.theta)[4]

    # ---------- прямой расчёт ----------
    def _forward(self, tau, e, s0, f, kap, inj, up):
        T = inj.shape[0]
        w = inj @ f + e                                   # [T,P]
        ue = up + kap * (1.0 - up)
        a = np.exp(-ue * self.step / tau)
        s = np.empty((T, self.P))
        s[0] = s0
        for k in range(1, T):
            s[k] = a[k] * s[k - 1] + (1.0 - a[k]) * w[k] / ue[k]
        return s, w, ue, a

    def simulate(self, inj, up, s_start=None, theta=None):
        """Свободный расчёт. s_start — состояние на первом шаге (по умолчанию обученное s0)."""
        tau, e, s0, f, kap = self._params(self.theta if theta is None else theta)
        if s_start is not None:
            s0 = s_start
        s, *_ = self._forward(tau, e, s0, f, kap, inj, up)
        return up * s, s

    # ---------- функция потерь и градиент ----------
    def _loss_grad(self, th, inj, up, q, wgt):
        tau, e, s0, f, kap = self._params(th)
        s, w, ue, a = self._forward(tau, e, s0, f, kap, inj, up)
        r = up * s - q
        loss = float((wgt * r * r).sum())
        g = 2.0 * wgt * r                                  # dL/dq_hat
        T = inj.shape[0]
        lam = np.zeros((T, self.P))                        # dL/ds_k
        lam[T - 1] = up[T - 1] * g[T - 1]
        for k in range(T - 2, -1, -1):
            lam[k] = up[k] * g[k] + a[k + 1] * lam[k + 1]
        dw = lam * (1.0 - a) / ue
        dw[0] = 0.0
        sprev = np.vstack([s[:1], s[:-1]])
        da = lam * (sprev - w / ue)
        da[0] = 0.0
        d_tau = (da * a * ue * self.step / tau ** 2).sum(axis=0)
        d_kap = (da * a * (-self.step / tau) * (1.0 - up)).sum() \
            + (lam * (1.0 - a) * w * (-(1.0 - up) / ue ** 2))[1:].sum()
        d_e = dw.sum(axis=0)
        d_f = inj.T @ dw                                   # [I,P]
        d_s0 = lam[0]

        a_, b_, c_, z, k_ = self._unpack(th)
        sg = _sigmoid(a_)
        g_a = d_tau * (self.tau_hi - self.tau_lo) * sg * (1 - sg)
        g_b = d_e * self.qs * _sigmoid(b_)
        g_c = d_s0 * self.qs * _sigmoid(c_)
        g_z = f * (d_f - (f * d_f).sum(axis=1, keepdims=True))
        sk = _sigmoid(k_)
        g_k = d_kap * (self.k_hi - self.k_lo) * sk * (1 - sk)

        zz = th[3 * self.P:3 * self.P + self.nz]
        loss += self.l2 * float((zz ** 2).sum())
        grad = np.concatenate([g_a, g_b, g_c, g_z[self.mask] + 2 * self.l2 * zz, [g_k]])
        return loss, grad

    # ---------- обучение ----------
    def _init_theta(self, inj, up, q):
        P = self.P
        act = up > 0.5
        qs = np.array([q[act[:, j], j].mean() if act[:, j].any() else 1.0 for j in range(P)])
        self.qs = np.maximum(qs, 1.0)
        tau0 = 30.0
        a = np.full(P, np.log((tau0 - self.tau_lo) / (self.tau_hi - tau0)))
        b = np.full(P, np.log(np.expm1(0.3)))               # e ~ 0.3 * qs
        first = np.array([q[act[:, j], j][:4].mean() if act[:, j].any() else self.qs[j] for j in range(P)])
        c = np.log(np.expm1(np.clip(first / self.qs, 0.05, 20)))
        # начальные f: обратно пропорционально расстоянию, сумма по добывающим 0.5
        wgt = np.where(self.mask, 1.0 / np.maximum(self.dist, 50.0), 0.0)
        ssum = wgt.sum(axis=1, keepdims=True)
        f0 = np.where(ssum > 0, 0.5 * wgt / np.maximum(ssum, 1e-12), 0.0)
        z = np.log(np.maximum(f0, 1e-6) / 0.5)              # при «пустом» выходе = 0.5
        return np.concatenate([a, b, c, z[self.mask], [0.0]])

    def fit(self, inj, up, q, maxiter=2000, theta0=None, half_life=None):
        """inj[T,I], up[T,P], q[T,P] — обучающий период. Потери нормированы на масштаб скважины.

        half_life (сут) — если задан, свежие данные весят больше: вес убывает вдвое за half_life.
        theta0 — «тёплый» старт от ранее обученной модели того же участка.
        """
        th0 = self._init_theta(inj, up, q)
        if theta0 is not None:
            th0 = theta0
        T = q.shape[0]
        tw = np.ones(T) if half_life is None else 0.5 ** ((T - 1 - np.arange(T)) * self.step / half_life)
        n_obs = np.maximum((tw[:, None] * (up > 0)).sum(axis=0), 1e-6)
        wgt = tw[:, None] * np.ones_like(q) / (n_obs * self.qs ** 2 * self.P)
        res = minimize(self._loss_grad, th0, args=(inj, up, q, wgt), jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': 1e-11, 'gtol': 1e-9})
        self.theta = res.x
        self.loss_ = res.fun
        self.nit_ = res.nit
        qh, _ = self.simulate(inj, up)
        act = up > 0.5
        self.fit_err_ = np.sqrt(((qh - q) ** 2 * act).sum(axis=0) / np.maximum(act.sum(axis=0), 1)) / self.qs
        return self

    # ---------- прогноз ----------
    def forecast(self, inj_hist, up_hist, inj_fut, up_fut, q_hist=None, anchor_steps=4):
        """Прогноз на будущий период.

        Без q_hist состояние на начало прогноза берётся из свободного расчёта на истории.
        С q_hist состояние «привязывается» к факту: s = средний факт. дебит работавшей
        скважины за последние anchor_steps шагов (простая ассимиляция данных).
        """
        _, s_hist = self.simulate(inj_hist, up_hist)
        s_last = s_hist[-1].copy()
        if q_hist is not None:
            qh, uh = q_hist[-anchor_steps:], up_hist[-anchor_steps:]
            ok = uh > 0.5
            cnt = ok.sum(axis=0)
            obs = np.where(ok, qh / np.maximum(uh, 1e-6), 0.0).sum(axis=0) / np.maximum(cnt, 1)
            s_last = np.where(cnt >= 2, obs, s_last)
        inj2 = np.vstack([inj_hist[-1:], inj_fut])
        up2 = np.vstack([up_hist[-1:], up_fut])
        q, s = self.simulate(inj2, up2, s_start=s_last)
        return q[1:], s[1:]
