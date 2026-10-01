import numpy as np

from simulation import init_spins, magnetization, metropolis_sweep, total_energy


def test_aligned_state_energy_is_minus_3NJ():
    # each site has 3 forward bonds, each contributing -J when aligned
    L = 4
    spins = np.zeros((L, L, L, 3))
    spins[..., 2] = 1.0
    assert np.isclose(total_energy(spins, L, 1.0), -3 * L**3)


def test_spins_stay_unit_vectors():
    L = 4
    spins = init_spins(L, seed=0)
    for _ in range(50):
        metropolis_sweep(spins, L, 1.5, 1.0)
    assert np.allclose(np.linalg.norm(spins, axis=-1), 1.0)


def test_orders_well_below_tc():
    L = 6
    spins = init_spins(L, seed=0)
    for _ in range(2000):
        metropolis_sweep(spins, L, 0.3, 1.0)
    assert np.linalg.norm(magnetization(spins)) / L**3 > 0.8


def test_disordered_well_above_tc():
    L = 6
    spins = init_spins(L, seed=0)
    for _ in range(500):
        metropolis_sweep(spins, L, 10.0, 1.0)
    assert np.linalg.norm(magnetization(spins)) / L**3 < 0.25
    