from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, minimize_scalar


@dataclass
class Properties:
    cp: float
    cw: float
    cs: float
    hfg: float


def properties_from_coolprop(P_boiling, T2_guess, fluid="Water"):
    import CoolProp.CoolProp as CP
    hf = CP.PropsSI('H', 'P', P_boiling, 'Q', 0, fluid)
    hg = CP.PropsSI('H', 'P', P_boiling, 'Q', 1, fluid)
    hfg = hg - hf
    cw = CP.PropsSI('C', 'P', P_boiling, 'Q', 0, fluid)
    cs_sat = CP.PropsSI('C', 'P', P_boiling, 'Q', 1, fluid)
    cs_T2 = CP.PropsSI('C', 'P', P_boiling, 'T', T2_guess, fluid)
    cs_bar = 0.5 * (cs_sat + cs_T2)
    return hfg, cw, cs_bar


def solve_state(M, N, tauH, taub, tau1, Ub_Us, Uw_Us, props: Properties, T0):
    cp, cw, cs, hfg = props.cp, props.cw, props.cs, props.hfg

    mu = M * cs / cp
    mup = mu * cw / cs

    def area_residual(tau2):
        eps_s = (tau2 - taub) / (tauH - taub)
        if not (0.0 < eps_s < 1.0):
            return None
        arg = (1 - eps_s) / (1 - eps_s * mu)
        if arg <= 0:
            return None
        Ns = -np.log(arg) / (1 - mu)
        x = mu * Ns / N

        tau3 = tauH - mu * eps_s * (tauH - taub)

        tau4 = tau3 - mu * hfg / (cs * T0)
        if tau4 <= taub:
            return None
        eps_b = (tau3 - tau4) / (tau3 - taub)
        if not (0.0 < eps_b < 1.0):
            return None
        Nb = -np.log(1 - eps_b)
        y = (1.0 / Ub_Us) * Nb / N

        eps_w = (taub - tau1) / (tau4 - tau1)
        if not (0.0 < eps_w < 1.0):
            return None
        arg2 = (1 - eps_w) / (1 - eps_w * mup)
        if arg2 <= 0:
            return None
        Nw = -np.log(arg2) / (1 - mup)
        aw_from_preheat = mup * (1.0 / Uw_Us) * Nw / N

        residual = aw_from_preheat - (1 - x - y)
        state = dict(x=x, y=y, tau2=tau2, tau3=tau3, tau4=tau4,
                     eps_s=eps_s, eps_b=eps_b, eps_w=eps_w, mup=mup)
        return residual, state

    lo, hi = taub * 1.0001, tauH * 0.9999
    scan = np.linspace(lo, hi, 400)

    def f(t2):
        out = area_residual(t2)
        return out[0] if out is not None else np.nan

    vals = [f(t) for t in scan]
    root = None
    for i in range(len(scan) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if a * b < 0:
            root = brentq(f, scan[i], scan[i + 1])
            break
    if root is None:
        return None

    _, st = area_residual(root)
    tau2, tau3, tau4 = st['tau2'], st['tau3'], st['tau4']
    eps_w, mup = st['eps_w'], st['mup']

    tauout = (1 - mup * eps_w) * tau4 + mup * eps_w * tau1

    a_term = tau2 - taub - np.log(tau2 / taub)
    b_term = (hfg / (cs * T0)) * (1 - 1 / taub) + (cw / cs) * (
        taub - tau1 - np.log(taub / tau1))
    # NOTE: use mu (= mw*cs/(m*cp), the capacity-rate ratio), NOT M (= mw/m).
    # W = mw*cs*T0*(a+b) and m*ex_H = m*cp*T0*(tauH-1-ln tauH), so
    # eta_II = (mw*cs)/(m*cp) * (a+b)/denom = mu*(a+b)/denom.
    # Using M here (as Eq. 34 of the paper literally states) leaves a
    # spurious extra factor of cp/cs in the result, which makes eta_II
    # scale artificially with the (unstated) hot-gas cp instead of being
    # cp-independent as a second-law efficiency should be.
    eta_II = mu * (a_term + b_term) / (tauH - 1 - np.log(tauH))

    return dict(tau2=tau2, tau3=tau3, tau4=tau4, tauout=tauout,
                x=st['x'], y=st['y'], aw=1 - st['x'] - st['y'],
                eps_s=st['eps_s'], eps_b=st['eps_b'], eps_w=eps_w,
                eta_II=eta_II, mu=mu, mup=mup)


def optimize_M(N, tauH, taub, tau1, Ub_Us, Uw_Us, props: Properties, T0,
               M_bounds=(0.01, 0.5)):

    def eta_of(M):
        r = solve_state(M, N, tauH, taub, tau1, Ub_Us, Uw_Us, props, T0)
        return r['eta_II'] if r is not None else np.nan

    def neg_eta(M):
        v = eta_of(M)
        return 1e6 if np.isnan(v) else -v   # sentinel ONLY for the bounded search

    Ms = np.linspace(M_bounds[0], M_bounds[1], 200)
    etas = np.array([eta_of(M) for M in Ms])   # <-- NaN for infeasible points, no sentinel

    if np.all(np.isnan(etas)):
        raise RuntimeError("No feasible M found in the given bounds.")

    i_best = np.nanargmax(etas)
    lo = Ms[max(i_best - 2, 0)]
    hi = Ms[min(i_best + 2, len(Ms) - 1)]

    res = minimize_scalar(neg_eta, bounds=(lo, hi), method='bounded')
    M_opt = res.x
    state = solve_state(M_opt, N, tauH, taub, tau1, Ub_Us, Uw_Us, props, T0)
    return M_opt, state, Ms, etas