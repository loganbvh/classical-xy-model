from typing import Optional, Tuple

import numba
import numpy as np


@numba.njit(fastmath=True)
def calculate_magnetization(phases: np.ndarray) -> float:
    Mx = np.sum(np.cos(phases))
    My = np.sum(np.sin(phases))
    M = np.sqrt(Mx**2 + My**2) / phases.size
    return M


@numba.njit(fastmath=True)
def calculate_stiffness(phases: np.ndarray) -> Tuple[float, float, float, float]:
    nrows, ncols = phases.shape
    N = phases.size
    ex = 0.0
    sx = 0.0
    ey = 0.0
    sy = 0.0
    for i in range(nrows):
        for j in range(ncols):
            delta_x = phases[i, j] - phases[i, (j - 1) % ncols]
            delta_y = phases[i, j] - phases[(i - 1) % nrows, j]
            ex += np.cos(delta_x)
            sx += np.sin(delta_x)
            ey += np.cos(delta_y)
            sy += np.sin(delta_y)
    ex /= N
    sx /= N
    ey /= N
    sy /= N
    return ex, sx, ey, sy


@numba.njit(fastmath=True)
def get_neighbors(
    row: int, col: int, nrows: int, ncols: int, periodic: bool = False
) -> Tuple[int, int, int, int, float, float, float, float]:
    if periodic:
        col_right_scale = col_left_scale = row_up_scale = row_down_scale = 1.0
        col_right = (col + 1) % ncols
        col_left = (col - 1) % ncols
        row_up = (row + 1) % nrows
        row_down = (row - 1) % nrows
    else:
        col_right = (col + 1) * (col < ncols - 1)
        col_left = (col - 1) * (col > 0)
        row_up = (row - 1) * (row > 0)
        row_down = (row + 1) * (row < nrows - 1)
        col_right_scale = float(int(col < ncols - 1))
        col_left_scale = float(int(col > 0))
        row_up_scale = float(int(row > 0))
        row_down_scale = float(int(row < nrows - 1))
    return (
        col_right,
        col_left,
        row_up,
        row_down,
        col_right_scale,
        col_left_scale,
        row_up_scale,
        row_down_scale,
    )


@numba.njit(fastmath=True)
def compute_trial_E(
    phases: np.ndarray,
    row: int,
    col: int,
    phase: Optional[float] = None,
    periodic: bool = False,
) -> float:
    nrows, ncols = phases.shape
    (
        col_right,
        col_left,
        row_up,
        row_down,
        col_right_scale,
        col_left_scale,
        row_up_scale,
        row_down_scale,
    ) = get_neighbors(row, col, nrows, ncols, periodic=periodic)

    if phase is None:
        phase = phases[row, col]

    E = -(
        row_up_scale * np.cos(phase - phases[row_up, col])
        + row_down_scale * np.cos(phase - phases[row_down, col])
        + col_right_scale * np.cos(phase - phases[row, col_right])
        + col_left_scale * np.cos(phase - phases[row, col_left])
    )
    return E


@numba.njit(fastmath=True)
def run_step(
    phases: np.ndarray, temperature: float, periodic: bool = False
) -> np.ndarray:
    nrows, ncols = phases.shape
    row = np.random.randint(0, nrows)
    col = np.random.randint(0, ncols)

    trial_phase = phases[row, col] + 0.5 * np.pi * (np.random.random() - 0.5)
    trial_phase = (trial_phase + np.pi) % (2 * np.pi) - np.pi

    # calculate E_new - E
    E_new = compute_trial_E(phases, row, col, periodic=periodic, phase=trial_phase)
    E_old = compute_trial_E(phases, row, col, periodic=periodic)
    delta_E = E_new - E_old

    # Metropolis update
    if (delta_E < 0) or (np.random.random() <= np.exp(-delta_E / temperature)):
        phases[row, col] = trial_phase

    return phases


@numba.njit(fastmath=True)
def calculate_E(phases: np.ndarray, periodic: bool = False) -> float:
    """
    Calculate energy per site
    """
    nrows, ncols = phases.shape
    E = 0.0
    if periodic:
        for row in range(nrows):
            for col in range(ncols):
                phase = phases[row, col]
                E -= (
                    np.cos(phase - phases[(row - 1) % nrows, col])
                    + np.cos(phase - phases[(row + 1) % nrows, col])
                    + np.cos(phase - phases[row, (col - 1) % ncols])
                    + np.cos(phase - phases[row, (col + 1) % ncols])
                )
    else:
        for row in range(1, nrows - 1):
            for col in range(1, ncols - 1):
                phase = phases[row, col]
                E -= (
                    np.cos(phase - phases[row - 1, col])
                    + np.cos(phase - phases[row + 1, col])
                    + np.cos(phase - phases[row, col - 1])
                    + np.cos(phase - phases[row, col + 1])
                )
    return 0.5 * E / phases.size


@numba.njit(fastmath=True)
def calculate_observables(
    nrows: int,
    ncols: int,
    temperature: float,
    hot_start: bool = False,
    thermalize: int = 100_000,
    samples: int = 1_000_000,
    periodic: bool = False,
) -> Tuple[np.ndarray, float, float, float, float, float]:

    if hot_start:
        phases = 2 * np.pi * (np.random.random((nrows, ncols)) - 0.5)
    else:
        phases = np.zeros((nrows, ncols), dtype=float)

    N = phases.size

    for _ in range(thermalize):
        phases = run_step(phases, temperature, periodic=periodic)

    energy = 0.0
    magnetization = 0.0
    magnetization2 = 0.0
    stiffness_x_e = 0.0
    stiffness_x_s2 = 0.0
    stiffness_y_e = 0.0
    stiffness_y_s2 = 0.0
    for _ in range(samples):
        phases = run_step(phases, temperature, periodic=periodic)
        M = calculate_magnetization(phases)
        magnetization += M
        magnetization2 += M**2
        energy += calculate_E(phases, periodic=periodic)
        ex, sx, ey, sy = calculate_stiffness(phases)
        stiffness_x_e += ex
        stiffness_x_s2 += sx**2
        stiffness_y_e += ey
        stiffness_y_s2 += sy**2

    magnetization /= samples
    magnetization2 /= samples
    stiffness_x_e /= samples
    stiffness_x_s2 /= samples
    stiffness_y_e /= samples
    stiffness_y_s2 /= samples
    energy /= samples
    susceptibility = (magnetization2 - magnetization**2) * N / temperature
    phase_stiffness_x = stiffness_x_e - N / temperature * stiffness_x_s2
    phase_stiffness_y = stiffness_y_e - N / temperature * stiffness_y_s2

    return (
        phases,
        energy,
        magnetization,
        susceptibility,
        phase_stiffness_x,
        phase_stiffness_y,
    )
