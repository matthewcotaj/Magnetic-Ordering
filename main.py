import os
import numpy as np
import matplotlib.pyplot as plt

from simulation import run_temperature_sweep
from analysis import (
    specific_heat, susceptibility, binder_cumulant, curie_weiss_fit,
    tc_from_binder_crossing, tc_from_susceptibility_peak,
)

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

L_VALUES = [6, 8, 10]
T_LIST = np.linspace(2.4, 0.8, 24)  # descending: anneal from disordered to ordered


def main():
    results = []
    for L in L_VALUES:
        print(f"L = {L} ({L**3} spins)")
        r = run_temperature_sweep(L, T_LIST, n_equil=2000, n_sample=5000, seed=1)
        results.append(r)
        np.savez(os.path.join(OUT_DIR, f"L{L}.npz"), **r)

    tc_est = tc_from_susceptibility_peak(results[-1])
    print(f"Tc estimate (chi peak, L={results[-1]['L']}): {tc_est:.3f} J/kB")

    tc_binder = tc_from_binder_crossing(results)
    if tc_binder:
        print(f"Tc estimate (Binder crossing, noisier at this L/statistics): {tc_binder:.3f} J/kB")
    else:
        print("Binder curves did not cross cleanly at this system size/statistics")

    plot_magnetization(results, tc_est)
    plot_susceptibility(results, tc_est)
    plot_specific_heat(results, tc_est)
    plot_binder(results, tc_est)
    plot_curie_weiss(results[-1], tc_est)


def plot_magnetization(results, tc_est):
    plt.figure(figsize=(6, 4.5))
    for r in results:
        m = r["M"] / r["N"]
        plt.plot(r["T"], m, "o-", label=f"L={r['L']}")
    if tc_est:
        plt.axvline(tc_est, color="gray", ls="--", lw=1, label=f"Tc ~ {tc_est:.2f}")
    plt.xlabel("T (J/kB)")
    plt.ylabel("|M| / N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "magnetization.png"), dpi=150)
    plt.close()


def plot_susceptibility(results, tc_est):
    plt.figure(figsize=(6, 4.5))
    for r in results:
        plt.plot(r["T"], susceptibility(r), "o-", label=f"L={r['L']}")
    if tc_est:
        plt.axvline(tc_est, color="gray", ls="--", lw=1)
    plt.xlabel("T (J/kB)")
    plt.ylabel("chi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "susceptibility.png"), dpi=150)
    plt.close()


def plot_specific_heat(results, tc_est):
    plt.figure(figsize=(6, 4.5))
    for r in results:
        plt.plot(r["T"], specific_heat(r), "o-", label=f"L={r['L']}")
    if tc_est:
        plt.axvline(tc_est, color="gray", ls="--", lw=1)
    plt.xlabel("T (J/kB)")
    plt.ylabel("C / N kB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "specific_heat.png"), dpi=150)
    plt.close()


def plot_binder(results, tc_est):
    plt.figure(figsize=(6, 4.5))
    for r in results:
        plt.plot(r["T"], binder_cumulant(r), "o-", label=f"L={r['L']}")
    if tc_est:
        plt.axvline(tc_est, color="gray", ls="--", lw=1, label=f"Tc ~ {tc_est:.2f}")
    plt.xlabel("T (J/kB)")
    plt.ylabel("Binder cumulant U_L")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "binder_cumulant.png"), dpi=150)
    plt.close()


def plot_curie_weiss(result, tc_est):
    T = result["T"]
    chi = susceptibility(result)
    T_min = (tc_est or T.mean()) * 1.3
    C, theta = curie_weiss_fit(T, chi, T_min)
    print(f"Curie-Weiss fit (L={result['L']}, T > {T_min:.2f}): C = {C:.3f}, theta = {theta:.3f}")

    mask = T > T_min
    plt.figure(figsize=(6, 4.5))
    plt.plot(T, 1 / chi, "o", label="1/chi (simulation)")
    T_fit = np.linspace(T_min, T.max(), 50)
    plt.plot(T_fit, (T_fit - theta) / C, "-", label=f"fit: theta={theta:.2f}, C={C:.2f}")
    plt.xlabel("T (J/kB)")
    plt.ylabel("1 / chi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "curie_weiss.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
