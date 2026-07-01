import numpy as np
import pyvinecopulib as pv

_FAMILY_MAP = {
    'N': pv.BicopFamily.gaussian,
    'T': pv.BicopFamily.student,
    'C': pv.BicopFamily.clayton,
    'F': pv.BicopFamily.frank,
    'G': pv.BicopFamily.gumbel,
}


def copulafitall3(copula_list, c, u):
    """
    copula_list: list of family names, e.g. ['N','T','C','F','G']
    c: selected copula name, e.g. 'C'
    u: (n, 2) array of uniform marginals
    returns: (value, tau)
        tau: Kendall's tau（pyvinecopulib 直接給，跨族可比）
    """
    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    try:
        bc = pv.Bicop(family=_FAMILY_MAP[c])
        bc.fit(u)
        value = bc.pdf(u)
        value = np.where(np.isfinite(value), value, 1e-300)
        value = np.maximum(value, 1e-300)
    except Exception:
        return np.full(len(u), 1e-300), 0.0

    return value, float(bc.tau)
