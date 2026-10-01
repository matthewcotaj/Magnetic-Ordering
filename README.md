# Magnetic Ordering Monte Carlo

Classical Heisenberg model on a 3D cubic lattice, simulated with Metropolis Monte Carlo to find the ferromagnetic ordering temperature.

Each site holds a 3D spin (unit vector), coupled to its 6 nearest neighbors with strength J. The simulation anneals the lattice from high to low temperature, and at each step proposes a random reorientation of a spin and accepts or rejects it with the Boltzmann factor. Averaging energy and magnetization at each temperature gives specific heat, susceptibility, and the Binder cumulant, which are the standard tools for pinning down a phase transition from finite-size simulation data.

## Why

My undergrad research was on lanthanide-doped nanorods (EuGdCuS3), where Gd3+ substitution changes the magnetic ordering behavior. That work was all synthesis and characterization; this is the finite-temperature theory side of the same physics, applied to a simpler model system I can actually simulate from scratch.

## Running it

```
pip install -r requirements.txt
python main.py
```

Takes about 1.5 minutes on a laptop. Runs 3 lattice sizes (6, 8, 10) across a temperature range and writes plots + raw data to `output/`.

## What it produces

- `magnetization.png`: order parameter dropping through the transition
- `susceptibility.png`: susceptibility peak, sharpening and growing with lattice size, which is the finite-size signature of a real phase transition
- `specific_heat.png`: specific heat vs T
- `binder_cumulant.png`: Binder cumulant curves for each L, used to locate Tc independent of system size
- `curie_weiss.png`: 1/chi fit to a Curie-Weiss law in the paramagnetic regime

Tc comes out around 1.50 J/kB from the susceptibility peak, close to the known value for this model (~1.44 J/kB), which is a decent sanity check that the simulation is doing what it should. The Binder cumulant crossing gives a consistent number but needs a lot more statistics than a quick run provides to be clean, so I'm using it as a secondary check rather than the primary estimate.

## Live view

```
python live_view.py
```

Opens a window with a live spin slice (arrows for the in-plane component, color for out-of-plane) next to a running magnetization plot, and sliders for T and J you can drag while it's running. Good for building intuition on how the system orders and disorders, not for generating the actual Tc numbers, that's what `main.py` is for. Needs a local display, won't work over SSH without X forwarding.

## Structure

- `simulation.py`: lattice setup and the Metropolis sweep (numba-jitted, otherwise this is too slow to be useful)
- `analysis.py`: specific heat, susceptibility, Binder cumulant, Curie-Weiss fit, Tc estimators
- `main.py`: runs the temperature sweep across lattice sizes and makes the plots
- `live_view.py`: interactive real-time view of the lattice with adjustable T and J

## Notes

Units are J = kB = 1 throughout, standard for this kind of model. Swapping in real exchange coupling and comparing directly against SQUID data would be the natural next step.

## Live demo

Interactive browser version (live lattice + temperature sweep): https://matthewcotaj.github.io/Website/projects/magnetism/index.html
