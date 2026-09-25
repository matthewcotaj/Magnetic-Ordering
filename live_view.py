import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from simulation import init_spins, metropolis_sweep, magnetization

L = 10
HISTORY = 1000

spins = init_spins(L, seed=1)
sweep_count = 0
mag_history = []

fig = plt.figure(figsize=(10, 5))
ax_lattice = fig.add_axes((0.06, 0.25, 0.4, 0.65))
ax_mag = fig.add_axes((0.56, 0.25, 0.4, 0.65))

X, Y = np.meshgrid(np.arange(L), np.arange(L))


def slice_uvc():
    s = spins[:, :, 0, :]  # z=0 slice
    return s[:, :, 0], s[:, :, 1], s[:, :, 2]


U, V, C = slice_uvc()
quiv = ax_lattice.quiver(X, Y, U, V, C, cmap="coolwarm", clim=(-1, 1), scale=15)
ax_lattice.set_xlim(-1, L)
ax_lattice.set_ylim(-1, L)
ax_lattice.set_aspect("equal")
ax_lattice.set_title("spin slice at z=0 (color = sz)")

(line,) = ax_mag.plot([], [])
ax_mag.set_xlim(0, HISTORY)
ax_mag.set_ylim(0, 1)
ax_mag.set_xlabel("sweep")
ax_mag.set_ylabel("|M| / N")
ax_mag.set_title("magnetization")

ax_T = fig.add_axes((0.15, 0.08, 0.3, 0.03))
ax_J = fig.add_axes((0.55, 0.08, 0.3, 0.03))
slider_T = Slider(ax_T, "T", 0.1, 3.0, valinit=1.5)
slider_J = Slider(ax_J, "J", 0.1, 2.0, valinit=1.0)


def update(frame):
    global sweep_count
    metropolis_sweep(spins, L, slider_T.val, slider_J.val)
    sweep_count += 1

    U, V, C = slice_uvc()
    quiv.set_UVC(U, V, C)

    m = np.linalg.norm(magnetization(spins)) / (L ** 3)
    mag_history.append(m)
    if len(mag_history) > HISTORY:
        mag_history.pop(0)
    xs = np.arange(sweep_count - len(mag_history) + 1, sweep_count + 1)
    line.set_data(xs, mag_history)
    ax_mag.set_xlim(xs[0], max(xs[-1], HISTORY))

    return quiv, line


if __name__ == "__main__":
    ani = FuncAnimation(fig, update, interval=40, cache_frame_data=False)
    plt.show()
