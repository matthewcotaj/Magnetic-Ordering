import numpy as np
from numba import njit


@njit(cache=True)
def _random_unit_vector():
    v = np.random.normal(0.0, 1.0, 3)
    n = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return v / n


@njit(cache=True)
def _seed_numba(seed):
    # numba has its own RNG; np.random.seed from regular Python doesn't reach it
    np.random.seed(seed)


@njit(cache=True)
def _local_field(spins, i, j, k, L):
    ip, im = (i + 1) % L, (i - 1) % L
    jp, jm = (j + 1) % L, (j - 1) % L
    kp, km = (k + 1) % L, (k - 1) % L
    fx = (spins[ip, j, k, 0] + spins[im, j, k, 0] + spins[i, jp, k, 0]
          + spins[i, jm, k, 0] + spins[i, j, kp, 0] + spins[i, j, km, 0])
    fy = (spins[ip, j, k, 1] + spins[im, j, k, 1] + spins[i, jp, k, 1]
          + spins[i, jm, k, 1] + spins[i, j, kp, 1] + spins[i, j, km, 1])
    fz = (spins[ip, j, k, 2] + spins[im, j, k, 2] + spins[i, jp, k, 2]
          + spins[i, jm, k, 2] + spins[i, j, kp, 2] + spins[i, j, km, 2])
    return fx, fy, fz


@njit(cache=True)
def metropolis_sweep(spins, L, T, J):
    N = L * L * L
    for _ in range(N):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        k = np.random.randint(0, L)
        fx, fy, fz = _local_field(spins, i, j, k, L)
        sx, sy, sz = spins[i, j, k, 0], spins[i, j, k, 1], spins[i, j, k, 2]
        e_old = -J * (sx * fx + sy * fy + sz * fz)
        nx, ny, nz = _random_unit_vector()
        e_new = -J * (nx * fx + ny * fy + nz * fz)
        dE = e_new - e_old
        if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
            spins[i, j, k, 0] = nx
            spins[i, j, k, 1] = ny
            spins[i, j, k, 2] = nz
    return spins


@njit(cache=True)
def total_energy(spins, L, J):
    e = 0.0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                ip, jp, kp = (i + 1) % L, (j + 1) % L, (k + 1) % L
                sx, sy, sz = spins[i, j, k, 0], spins[i, j, k, 1], spins[i, j, k, 2]
                e += sx * spins[ip, j, k, 0] + sy * spins[ip, j, k, 1] + sz * spins[ip, j, k, 2]
                e += sx * spins[i, jp, k, 0] + sy * spins[i, jp, k, 1] + sz * spins[i, jp, k, 2]
                e += sx * spins[i, j, kp, 0] + sy * spins[i, j, kp, 1] + sz * spins[i, j, kp, 2]
    return -J * e


def magnetization(spins):
    m = spins.reshape(-1, 3).sum(axis=0)
    return m


def init_spins(L, seed=None):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(L, L, L, 3))
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v


def run_temperature_sweep(L, T_list, n_equil=2000, n_sample=4000, J=1.0, seed=None):
    """Anneal from high to low T, collecting equilibrium averages at each step."""
    N = L ** 3
    spins = init_spins(L, seed=seed)
    if seed is not None:
        _seed_numba(seed)

    E_avg, E2_avg = [], []
    M_avg, M2_avg, M4_avg = [], [], []

    for T in T_list:
        for _ in range(n_equil):
            metropolis_sweep(spins, L, T, J)

        E_samples = np.empty(n_sample)
        M_samples = np.empty(n_sample)
        for s in range(n_sample):
            metropolis_sweep(spins, L, T, J)
            E_samples[s] = total_energy(spins, L, J)
            M_samples[s] = np.linalg.norm(magnetization(spins))

        E_avg.append(E_samples.mean())
        E2_avg.append((E_samples ** 2).mean())
        M_avg.append(M_samples.mean())
        M2_avg.append((M_samples ** 2).mean())
        M4_avg.append((M_samples ** 4).mean())

    return {
        "L": L,
        "N": N,
        "T": np.array(T_list),
        "E": np.array(E_avg),
        "E2": np.array(E2_avg),
        "M": np.array(M_avg),
        "M2": np.array(M2_avg),
        "M4": np.array(M4_avg),
    }
