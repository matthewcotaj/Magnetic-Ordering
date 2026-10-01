import numpy as np


def specific_heat(result):
    T, N = result["T"], result["N"]
    return (result["E2"] - result["E"] ** 2) / (N * T ** 2)


def susceptibility(result):
    T, N = result["T"], result["N"]
    return (result["M2"] - result["M"] ** 2) / (N * T)

def susceptibility_full(result):
    # <M^2>/(N T): the right chi in the paramagnetic phase, where the
    # connected form subtracts a large finite-size <|M|>^2 and suppresses chi
    T, N = result["T"], result["N"]
    return result["M2"] / (N * T)


def binder_cumulant(result):
    return 1.0 - result["M4"] / (3.0 * result["M2"] ** 2)


def curie_weiss_fit(T, chi, T_min):
    """Fit 1/chi = (T - theta) / C over the paramagnetic tail, T > T_min."""
    mask = T > T_min
    inv_chi = 1.0 / chi[mask]
    slope, intercept = np.polyfit(T[mask], inv_chi, 1)
    C = 1.0 / slope
    theta = -intercept * C
    return C, theta


def tc_from_susceptibility_peak(result):
    # where chi is maximal for this L. Finite-size shifted, but robust.
    order = np.argsort(result["T"])
    T = result["T"][order]
    chi = susceptibility(result)[order]
    return float(T[np.argmax(chi)])


def tc_from_binder_crossing(results, slope_frac=0.3):
    # curves flatten out away from Tc and overlap almost exactly there, so a
    # plain sign-change search just picks up noise. Only look where they're
    # actually moving.
    T_fine = np.linspace(
        max(r["T"].min() for r in results),
        min(r["T"].max() for r in results),
        2000,
    )
    curves = []
    for r in results:
        order = np.argsort(r["T"])
        curves.append(np.interp(T_fine, r["T"][order], binder_cumulant(r)[order]))

    avg_curve = np.mean(curves, axis=0)
    slope = np.abs(np.gradient(avg_curve, T_fine))
    active = slope >= slope_frac * slope.max()

    crossings = []
    for a in range(len(curves)):
        for b in range(a + 1, len(curves)):
            diff = (curves[a] - curves[b])[active]
            T_active = T_fine[active]
            sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
            crossings.extend(T_active[sign_change])

    if not crossings:
        return None
    return float(np.mean(crossings))
