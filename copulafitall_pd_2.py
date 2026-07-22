import numpy as np
import pyvinecopulib as pv

_FAMILY_MAP = {
    'copulaN': pv.BicopFamily.gaussian,
    'copulaT': pv.BicopFamily.student,
    'copulaC': pv.BicopFamily.clayton,
    'copulaF': pv.BicopFamily.frank,
    'copulaG': pv.BicopFamily.gumbel,
}


def copulafitall23(copula_list, c, u):
    """
    copula_list: list of family names, e.g. ['copulaN','copulaT','copulaC','copulaF','copulaG']
    c: selected copula name, e.g. 'copulaC'
    u: (n, 2) array of uniform marginals
    returns: (value, tau)
        tau: Kendall's tau（pyvinecopulib 直接給，跨族可比）
    """
    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    try:
        bc = pv.Bicop.from_data(
            u, controls=pv.FitControlsBicop(
                family_set=[_FAMILY_MAP[c]], preselect_families=False
            )
        )
        value = bc.pdf(u)
        value = np.where(np.isfinite(value), value, 1e-300)
        value = np.maximum(value, 1e-300)
    except Exception:
        return np.full(len(u), 1e-300), 0.0

    return value, float(bc.tau)
