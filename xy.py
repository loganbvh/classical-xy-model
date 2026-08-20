import dataclasses
import multiprocessing as mp
from datetime import datetime
from functools import partial
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numba
import numpy as np
from tqdm import tqdm
from uncertainties import UFloat, ufloat, ufloat_fromstr


@dataclasses.dataclass
class XYResult:
    """A container for the results of a Monte Carlo simulation at a given temperature."""

    temperature: float
    J: float
    nrows: int
    ncols: int
    hot_start: bool
    metropolis_steps_per_pass: int
    thermalize_passes: int
    measure_passes: int
    periodic: bool
    rng_seed: Union[int, None]
    energy: UFloat
    magnetization: UFloat
    susceptibility: UFloat
    helicity_x: UFloat
    helicity_y: UFloat
    algorithm: Literal["wolff", "metropolis"]
    start_time: datetime
    end_time: datetime
    total_seconds: float

    def to_json(self) -> Dict[str, Union[int, float, str, None]]:
        """Convert the ``XYResult`` to a JSON-compatible dict."""
        json_dict = {}
        for key, value in dataclasses.asdict(self).items():
            if isinstance(value, UFloat):
                value = repr(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            json_dict[key] = value
        return json_dict

    @staticmethod
    def from_json(
        json_dict: Dict[str, Union[int, float, str, UFloat, datetime, None]],
    ) -> "XYResult":
        """Create an ``XYResult`` from a JSON-compatible dict"""
        kwargs = {}
        for field in dataclasses.fields(XYResult):
            try:
                value = json_dict[field.name]
            except KeyError:
                continue
            if isinstance(value, str):
                if "+/-" in value:
                    value = ufloat_fromstr(value)
                try:
                    value = datetime.fromisoformat(value)
                except:
                    pass
            kwargs[field.name] = value
        return XYResult(**kwargs)


@numba.njit
def get_site_neighbors_scales(
    row: int, col: int, nrows: int, ncols: int, periodic: bool = False
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    """Returns the four nearest neighbor indices for a given site, and a scale
    value (either 0 or 1) indicating whether to include the corresponding bond.
    """
    col_right = (col + 1) % ncols
    col_left = (col - 1) % ncols
    row_up = (row - 1) % nrows
    row_down = (row + 1) % nrows

    if periodic:
        col_right_scale = col_left_scale = row_up_scale = row_down_scale = 1.0
    else:
        col_right_scale = col < ncols - 1
        col_left_scale = col > 0
        row_up_scale = row > 0
        row_down_scale = row < nrows - 1

    neighbors = col_right, col_left, row_up, row_down
    scales = col_right_scale, col_left_scale, row_up_scale, row_down_scale
    return neighbors, scales


@numba.njit
def get_all_neighbors_scales(
    nrows: int, ncols: int, periodic: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the four nearest neighbor indices for a all sites, and a scale
    value (either 0 or 1) indicating whether to include the corresponding bond.
    """
    # (nrows, ncols, [col_right, col_left, row_up, row_down])
    neighbors = np.empty((nrows, ncols, 4), dtype=np.int64)
    scales = np.empty((nrows, ncols, 4), dtype=float)

    for row in range(nrows):
        for col in range(ncols):
            site_neighbors, site_scales = get_site_neighbors_scales(
                row, col, nrows, ncols, periodic=periodic
            )
            neighbors[row, col] = site_neighbors
            scales[row, col] = site_scales
    return neighbors, scales


@numba.njit(fastmath=True)
def calculate_site_energy(
    phases: np.ndarray,
    neighbors: np.ndarray,
    scales: np.ndarray,
    row: int,
    col: int,
    site_phase: float,
    J: float = 1.0,
) -> float:
    """Calculates the total energy of a given site with phase ``site_phase``."""

    col_right, col_left, row_up, row_down = neighbors[row, col]
    col_right_scale, col_left_scale, row_up_scale, row_down_scale = scales[row, col]

    E = -J * (
        col_right_scale * np.cos(site_phase - phases[row, col_right])
        + col_left_scale * np.cos(site_phase - phases[row, col_left])
        + row_up_scale * np.cos(site_phase - phases[row_up, col])
        + row_down_scale * np.cos(site_phase - phases[row_down, col])
    )
    return E


@numba.njit(fastmath=True)
def run_n_metropolis_steps(
    n: int,
    phases: np.ndarray,
    neighbors: np.ndarray,
    scales: np.ndarray,
    temperature: float,
    J: float = 1.0,
) -> np.ndarray:
    """Performs ``n`` Metropolis updates and returns the resulting phases."""
    nrows, ncols = phases.shape

    for _ in range(n):
        row = np.random.randint(0, nrows)
        col = np.random.randint(0, ncols)

        trial_phase = phases[row, col] + np.pi * (np.random.random() - 0.5)
        trial_phase = (trial_phase + np.pi) % (2 * np.pi) - np.pi

        # calculate E_new - E
        E_new = calculate_site_energy(
            phases,
            neighbors,
            scales,
            row,
            col,
            site_phase=trial_phase,
            J=J,
        )
        E_old = calculate_site_energy(
            phases,
            neighbors,
            scales,
            row,
            col,
            site_phase=phases[row, col],
            J=J,
        )
        delta_E = E_new - E_old

        # Metropolis update
        if (delta_E < 0) or (np.random.random() < np.exp(-delta_E / temperature)):
            phases[row, col] = trial_phase

    return phases


@numba.njit(fastmath=True)
def calculate_energy(
    phases: np.ndarray,
    neighbors: np.ndarray,
    scales: np.ndarray,
    J: float = 1.0,
) -> float:
    """Calculates the average energy per site for a given configuration."""
    nrows, ncols = phases.shape
    E = 0.0
    for row in range(nrows):
        for col in range(ncols):
            E += calculate_site_energy(
                phases, neighbors, scales, row, col, phases[row, col]
            )
    return 0.5 * E / phases.size


@numba.njit(fastmath=True)
def calculate_magnetization(phases: np.ndarray) -> float:
    """Calculates the average magnetization per site for a given configuration."""
    Mx = np.sum(np.cos(phases))
    My = np.sum(np.sin(phases))
    M = np.sqrt(Mx**2 + My**2) / phases.size
    return M


@numba.njit(fastmath=True)
def calculate_helicity_periodic(
    phases: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Calculates the quantities needed to compute the helicity modulus
    for a given configuration (with periodic boundary conditions).
    """
    nrows, ncols = phases.shape
    ex = sx = ey = sy = 0.0
    for i in range(nrows):
        for j in range(ncols):
            delta_x = phases[i, j] - phases[i, (j + 1) % ncols]
            delta_y = phases[i, j] - phases[(i + 1) % nrows, j]
            ex += np.cos(delta_x)
            sx += np.sin(delta_x)
            ey += np.cos(delta_y)
            sy += np.sin(delta_y)
    return ex, sx, ey, sy


# @numba.njit(fastmath=True)
# def calculate_helicity_open(phases: np.ndarray) -> Tuple[float, float, float, float]:
#     """Calculates the quantities needed to compute the helicity modulus
#     for a given configuration (with open boundary conditions).
#     """

#     nrows, ncols = phases.shape
#     # stiffness_x
#     ex = 0.0
#     sx = 0.0
#     for i in range(nrows):
#         for j in range(ncols - 1):
#             delta_x = phases[i, j] - phases[i, j + 1]
#             ex += np.cos(delta_x)
#             sx += np.sin(delta_x)

#     # stiffness_y
#     ey = 0.0
#     sy = 0.0
#     for i in range(nrows - 1):
#         for j in range(ncols):
#             delta_y = phases[i, j] - phases[i + 1, j]
#             ey += np.cos(delta_y)
#             sy += np.sin(delta_y)

#     return ex, sx, ey, sy


@numba.njit(fastmath=True)
def construct_wolff_cluster(
    row: int,
    col: int,
    r0: complex,
    phases: np.ndarray,
    neighbors: np.ndarray,
    scales: np.ndarray,
    visited: np.ndarray,
    temperature: np.ndarray,
    J: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recursively constructs a cluster starting at a given site using the
    Wolff algorithm, flipping spins as the cluster grows.
    """

    S0 = np.exp(1j * phases[row, col])
    S0_dot_r0 = (S0 / r0).real

    col_right, col_left, row_up, row_down = neighbors[row, col]
    nearest_neighbors = np.array(
        [(row, col_right), (row, col_left), (row_up, col), (row_down, col)]
    )

    for k, (i, j) in enumerate(nearest_neighbors):
        if visited[i, j] or (scales[row, col, k] == 0):
            continue

        S = np.exp(1j * phases[i, j])
        S_dot_r0 = (S / r0).real

        a = 2 / temperature * S0_dot_r0 * S_dot_r0 * J
        prob = (a < 0) * (1 - np.exp(a))
        if np.random.random() < prob:
            S -= 2 * S_dot_r0 * r0
            phases[i, j] = np.angle(S)
            visited[i, j] = True

            phases, visited = construct_wolff_cluster(
                i,
                j,
                r0,
                phases,
                neighbors,
                scales,
                visited,
                temperature,
                J=J,
            )

    return phases, visited


@numba.njit(fastmath=True)
def run_wolff_step(
    phases: np.ndarray,
    neighbors: np.ndarray,
    scales: np.ndarray,
    temperature: float,
    J: float = 1.0,
) -> np.ndarray:
    """Performs a single Wolff update by constructing and flipping a cluster."""

    nrows, ncols = phases.shape
    visited = np.zeros((nrows, ncols), dtype=numba.boolean)

    # Pick initial site at random
    row = np.random.randint(0, nrows)
    col = np.random.randint(0, ncols)
    S = np.exp(1j * phases[row, col])
    r0 = np.exp(1j * 2 * np.pi * np.random.random())

    # Reflect spin at initial site aboout the plane perpendicular to r0
    S_dot_r0 = (S / r0).real
    S -= 2 * S_dot_r0 * r0
    phases[row, col] = np.angle(S)
    visited[row, col] = True

    # Build cluster starting from initial site
    phases, visited = construct_wolff_cluster(
        row,
        col,
        r0,
        phases,
        neighbors,
        scales,
        visited,
        temperature,
        J=J,
    )
    return phases


def run_model(
    temperature: float,
    J: float,
    *,
    nrows: int,
    ncols: int,
    hot_start: bool = False,
    metropolis_steps_per_pass: Optional[int] = None,
    thermalize_passes: int = 1_000,
    measure_passes: int = 10_000,
    periodic: bool = False,
    rng_seed: Optional[int] = None,
    progress_bar: bool = True,
    use_wolff: bool = True,
) -> XYResult:

    start_time = datetime.now()

    if rng_seed is not None:
        np.random.seed(rng_seed)

    if metropolis_steps_per_pass is None:
        metropolis_steps_per_pass = nrows * ncols

    if hot_start:
        phases = 2 * np.pi * (np.random.random((nrows, ncols)) - 0.5)
    else:
        phases = np.zeros((nrows, ncols), dtype=float)

    neighbors, scales = get_all_neighbors_scales(nrows, ncols, periodic=periodic)

    # Thermalize
    for _ in tqdm(
        range(thermalize_passes), desc="Thermalizing", disable=(not progress_bar)
    ):
        if use_wolff:
            phases = run_wolff_step(phases, neighbors, scales, temperature, J=J)
        else:
            phases = run_n_metropolis_steps(
                metropolis_steps_per_pass,
                phases,
                neighbors,
                scales,
                temperature,
                J=J,
            )

    _energy = np.zeros(measure_passes)
    _magnetization = np.zeros(measure_passes)
    _magnetization2 = np.zeros(measure_passes)
    _helicity_x_e = np.zeros(measure_passes)
    _helicity_x_s2 = np.zeros(measure_passes)
    _helicity_y_e = np.zeros(measure_passes)
    _helicity_y_s2 = np.zeros(measure_passes)

    for i in tqdm(range(measure_passes), desc="Measuring", disable=(not progress_bar)):
        if use_wolff:
            phases = run_wolff_step(phases, neighbors, scales, temperature, J=J)
        else:
            phases = run_n_metropolis_steps(
                metropolis_steps_per_pass,
                phases,
                neighbors,
                scales,
                temperature,
                J=J,
            )

        _energy[i] = calculate_energy(phases, neighbors, scales, J=J)
        M = calculate_magnetization(phases)
        _magnetization[i] = M
        _magnetization2[i] = M**2
        ex, sx, ey, sy = calculate_helicity_periodic(phases)
        _helicity_x_e[i] = ex
        _helicity_x_s2[i] = sx**2
        _helicity_y_e[i] = ey
        _helicity_y_s2[i] = sy**2

    energy = ufloat(np.mean(_energy), np.std(_energy))
    magnetization = ufloat(np.mean(_magnetization), np.std(_magnetization))
    magnetization2 = ufloat(np.mean(_magnetization2), np.std(_magnetization2))
    susceptibility = (magnetization2 - magnetization**2) * phases.size / temperature

    helicity_x_e = ufloat(np.mean(_helicity_x_e), np.std(_helicity_x_e))
    helicity_x_s2 = ufloat(np.mean(_helicity_x_s2), np.std(_helicity_x_s2))
    helicity_y_e = ufloat(np.mean(_helicity_y_e), np.std(_helicity_y_e))
    helicity_y_s2 = ufloat(np.mean(_helicity_y_s2), np.std(_helicity_y_s2))

    if periodic:
        Nx = Ny = phases.size
    else:
        Nx = nrows * (ncols - 1)
        Ny = (nrows - 1) * ncols

    helicity_x = J * (1 / Nx) * (helicity_x_e - J * helicity_x_s2 / temperature)
    helicity_y = J * (1 / Ny) * (helicity_y_e - J * helicity_y_s2 / temperature)

    end_time = datetime.now()

    result = XYResult(
        temperature=temperature,
        J=J,
        nrows=nrows,
        ncols=ncols,
        hot_start=hot_start,
        metropolis_steps_per_pass=metropolis_steps_per_pass,
        thermalize_passes=thermalize_passes,
        measure_passes=measure_passes,
        periodic=periodic,
        rng_seed=rng_seed,
        energy=energy,
        magnetization=magnetization,
        susceptibility=susceptibility,
        helicity_x=helicity_x,
        helicity_y=helicity_y,
        algorithm="wolff" if use_wolff else "metropolis",
        start_time=start_time,
        end_time=end_time,
        total_seconds=(end_time - start_time).total_seconds(),
    )

    return result


def run_temperature_sweep(
    *,
    temperatures: Sequence[float],
    num_cpus: int,
    nrows: int,
    ncols: int,
    hot_start: bool = False,
    metropolis_steps_per_pass: Optional[int] = None,
    thermalize_passes: int = 1_000,
    measure_passes: int = 10_000,
    periodic: bool = False,
    use_wolff: bool = True,
    progress_bar: bool = True,
    rng_seed: Optional[int] = None,
    Js: Optional[Sequence[float]] = None,
) -> List[XYResult]:

    func = partial(
        run_model,
        nrows=nrows,
        ncols=ncols,
        hot_start=hot_start,
        metropolis_steps_per_pass=metropolis_steps_per_pass,
        thermalize_passes=thermalize_passes,
        measure_passes=measure_passes,
        periodic=periodic,
        rng_seed=rng_seed,
        use_wolff=use_wolff,
        progress_bar=False,
    )

    if Js is None:
        Js = np.ones(len(temperatures), dtype=float)

    if num_cpus > 1:
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.starmap(func, list(zip(temperatures, Js)))
    else:
        results = []
        for T, J in tqdm(
            zip(temperatures, Js), desc="Temperatures", disable=(not progress_bar)
        ):
            results.append(func(T, J))

    return results
