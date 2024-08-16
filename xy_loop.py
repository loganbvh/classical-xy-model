import dataclasses
import json
import multiprocessing as mp
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import pint
import superscreen as sc
from scipy import special
from scipy.constants import mu_0
from tqdm import tqdm
from uncertainties import UFloat, ufloat, ufloat_fromstr

ureg = pint.UnitRegistry()


@dataclasses.dataclass
class XYResult:
    """A container for the results of a Monte Carlo simulation at a given temperature."""

    temperature: float
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
    squid_susceptibility: UFloat
    start_time: datetime
    end_time: datetime
    total_seconds: float
    J: float = 1.0

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
        json_dict: Dict[str, Union[int, float, str, UFloat, datetime, None]]
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


def current_loop_vector_potential(
    positions: np.ndarray,
    *,
    loop_center: Tuple[float] = (0, 0, 0),
    loop_radius: float = 1,
    current: float = 1,
    length_units: str = "um",
    current_units: str = "uA",
):
    """Calculates the magnetic vector potential [Ax, Ay] at ``positions``
    due to a 1D current loop.

    Args:
        positions: Shape (n, 3) array of (x, y, z) positions at which to
            evaluate the vector potential.
        loop_center: (x, y, z) coordinates of the current loop center.
        loop_radius: radius of the current loop.
        current: Magnitude of the current flowing in the loop.
        length_units: A string specifying the length units.
        current_units: A string specifying the current units.

    Returns:
        Shape (n, 3) array of the vector potential [Ax, Ay, Az] at ``positions``.
    """
    to_meter = ureg(length_units).to("m").magnitude
    to_amp = ureg(current_units).to("A").magnitude
    # http://www.physics.usu.edu/Wheeler/EMarchive/Jch5Notes.pdf
    positions = np.atleast_2d(positions) * to_meter
    loop_center = np.atleast_2d(loop_center) * to_meter
    a = loop_radius * to_meter
    current = current * to_amp
    positions = positions - loop_center
    # # This is a pint-friendly vector norm.
    # rs = np.sqrt(np.sum(np.square(positions), axis=1))
    rs = np.linalg.norm(positions, axis=1)
    thetas = np.arccos(positions[:, 2] / rs)
    sin_thetas = np.sin(thetas)
    # m == k**2, see docs for scipy.special.ellipk
    denom = rs**2 + a**2 + 2 * a * rs * sin_thetas
    m = 4 * a * rs * sin_thetas / denom
    K = special.ellipk(m)
    E = special.ellipe(m)
    mag = -mu_0 * current * a / (np.pi * m) * (((m - 2) * K + 2 * E)) / np.sqrt(denom)
    # \vec{A} is directed along the azimuthal direction,
    # so here we generate the azimuthal unit vector.
    # Azimuthal angle + pi / 2 to get azimuthal direction.
    phis = np.arctan2(positions[:, 1], positions[:, 0]) + np.pi / 2
    direc = np.array([np.cos(phis), np.sin(phis), np.zeros_like(phis)]).T
    return mag[:, np.newaxis] * direc * ureg("T * m")


def make_island_centers(nrows: int, ncols: int, lattice_constant: float) -> np.ndarray:
    x0 = y0 = 0
    width = lattice_constant * ncols
    height = lattice_constant * nrows
    xmin = x0 - width / 2
    ymin = y0 - height / 2
    xs = xmin + (lattice_constant / 2) + lattice_constant * np.arange(ncols)
    ys = ymin + (lattice_constant / 2) + lattice_constant * np.arange(nrows)
    X, Y = np.meshgrid(xs, ys)
    return np.array([X, Y]).transpose(1, 2, 0)


def get_vector_potential(
    loop_center: Tuple[float, float, float],
    loop_radius: float,
    loop_current: float,
    island_centers: np.ndarray,
) -> np.ndarray:
    nrows, ncols, _ = island_centers.shape
    xy = island_centers.reshape((-1, 2))
    xyz = np.append(xy, np.zeros((len(xy), 1)), axis=1)
    A = current_loop_vector_potential(
        xyz,
        loop_center=loop_center,
        loop_radius=loop_radius,
        current=loop_current,
        length_units="um",
        current_units="uA",
    )
    Axy = (2 * np.pi / ureg("Phi_0") * A).to("1 / um").magnitude[:, :2]
    return Axy.reshape((nrows, ncols, 2))


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
    J: float = 1.0,
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
            ex += J * np.cos(delta_x)
            sx += J * np.sin(delta_x)
            ey += J * np.cos(delta_y)
            sy += J * np.sin(delta_y)
    return ex, sx, ey, sy


@numba.njit(fastmath=True)
def calculate_helicity_open(
    phases: np.ndarray, J: float = 1.0
) -> Tuple[float, float, float, float]:
    """Calculates the quantities needed to compute the helicity modulus
    for a given configuration (with open boundary conditions).
    """

    nrows, ncols = phases.shape
    # stiffness_x
    ex = 0.0
    sx = 0.0
    for i in range(nrows):
        for j in range(ncols - 1):
            delta_x = phases[i, j] - phases[i, j + 1]
            ex += J * np.cos(delta_x)
            sx += J * np.sin(delta_x)

    # stiffness_y
    ey = 0.0
    sy = 0.0
    for i in range(nrows - 1):
        for j in range(ncols):
            delta_y = phases[i, j] - phases[i + 1, j]
            ey += J * np.cos(delta_y)
            sy += J * np.sin(delta_y)

    return ex, sx, ey, sy


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
    A: np.ndarray,
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

    A0 = A[row, col]
    A1 = 0.5 * (A0[0] + A[row, col_right, 0])
    A2 = 0.5 * (A0[0] + A[row, col_left, 0])
    A3 = 0.5 * (A0[1] + A[row_up, col, 1])
    A4 = 0.5 * (A0[1] + A[row_down, col, 1])

    E = -J * (
        col_right_scale * np.cos(phases[row, col_right] - site_phase - A1)
        + col_left_scale * np.cos(phases[row, col_left] - site_phase + A2)
        + row_up_scale * np.cos(phases[row_up, col] - site_phase - A3)
        + row_down_scale * np.cos(phases[row_down, col] - site_phase + A4)
    )
    return E


@numba.njit(fastmath=True)
def run_n_metropolis_steps(
    n: int,
    phases: np.ndarray,
    A: np.ndarray,
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
            phases, A, neighbors, scales, row, col, site_phase=trial_phase, J=J
        )
        E_old = calculate_site_energy(
            phases, A, neighbors, scales, row, col, site_phase=phases[row, col], J=J
        )
        delta_E = E_new - E_old

        # Metropolis update
        if (delta_E < 0) or (np.random.random() < np.exp(-delta_E / temperature)):
            phases[row, col] = trial_phase

    return phases


@numba.njit(fastmath=True)
def calculate_energy(
    phases: np.ndarray,
    A: np.ndarray,
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
                phases, A, neighbors, scales, row, col, phases[row, col], J=J
            )
    return 0.5 * E / phases.size


def get_currents(phases: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Ax = (A[1:, :-1, 0] + A[1:, 1:, 0]) / 2
    Ay = (A[:-1, 1:, 1] + A[1:, 1:, 1]) / 2

    delta_x = phases[:, 1:] - phases[:, :-1]  # right - left
    delta_y = phases[:-1, :] - phases[1:, :]  # top - bottom

    Ix = np.sin(delta_x[:-1, :] - Ax)
    Iy = np.sin(delta_y[:, :-1] - Ay)

    return Ix, Iy


@numba.njit(fastmath=True)
def biot_savart_from_lattice(
    eval_positions: np.ndarray,
    bond_positions: np.ndarray,
    bond_currents: np.ndarray,
) -> np.ndarray:

    assert eval_positions.ndim == 2
    assert eval_positions.shape[1] == 3
    assert bond_positions.ndim == 2
    assert bond_positions.shape[0] == bond_currents.shape[0]
    assert bond_positions.shape[1] == 3

    Ix = bond_currents[:, 0]
    Iy = bond_currents[:, 1]
    Bz_out = np.empty(len(eval_positions), dtype=float)

    for i in range(eval_positions.shape[0]):
        Ix_dy = 0.0
        Iy_dx = 0.0
        for k in range(bond_positions.shape[0]):
            dx = eval_positions[i, 0] - bond_positions[k, 0]
            dy = eval_positions[i, 1] - bond_positions[k, 1]
            dz = eval_positions[i, 2] - bond_positions[k, 2]
            pref = (mu_0 / (4 * np.pi)) * (dx * dx + dy * dy + dz * dz) ** (-3 / 2)
            Ix_dy += pref * Ix[k] * dy
            Iy_dx += pref * Iy[k] * dx
        Bz_out[i] = Ix_dy - Iy_dx
    return Bz_out


# @numba.njit(fastmath=True)
# def vector_potential_from_lattice(
#     eval_positions: np.ndarray,
#     bond_positions: np.ndarray,
#     bond_currents: np.ndarray,
# ) -> np.ndarray:

#     assert eval_positions.ndim == 2
#     assert eval_positions.shape[1] == 3
#     assert bond_positions.ndim == 2
#     assert bond_positions.shape[0] == bond_currents.shape[0]
#     assert bond_positions.shape[1] == 3

#     Ix = bond_currents[:, 0]
#     Iy = bond_currents[:, 1]
#     Axy_out = np.empty((len(eval_positions), 2), dtype=float)

#     for i in range(eval_positions.shape[0]):
#         Ax = 0.0
#         Ay = 0.0
#         for k in range(bond_positions.shape[0]):
#             dx = eval_positions[i, 0] - bond_positions[k, 0]
#             dy = eval_positions[i, 1] - bond_positions[k, 1]
#             dz = eval_positions[i, 2] - bond_positions[k, 2]
#             dr = (dx * dx + dy * dy + dz * dz) ** (1 / 2)
#             pref = mu_0 / (4 * np.pi) / dr
#             Ax += pref * Ix[k]
#             Ay += pref * Iy[k]
#         Axy_out[i, 0] = Ax
#         Axy_out[i, 1] = Ay
#     return Axy_out


# @numba.njit(fastmath=True)
# def construct_wolff_cluster(
#     row: int,
#     col: int,
#     r0: complex,
#     phases: np.ndarray,
#     neighbors: np.ndarray,
#     scales: np.ndarray,
#     visited: np.ndarray,
#     temperature: np.ndarray,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Recursively constructs a cluster starting at a given site using the
#     Wolff algorithm, flipping spins as the cluster grows.
#     """

#     S0 = np.exp(1j * phases[row, col])
#     S0_dot_r0 = (S0 / r0).real

#     col_right, col_left, row_up, row_down = neighbors[row, col]
#     nearest_neighbors = np.array(
#         [(row, col_right), (row, col_left), (row_up, col), (row_down, col)]
#     )

#     for k, (i, j) in enumerate(nearest_neighbors):
#         if visited[i, j] or (scales[row, col, k] == 0):
#             continue

#         S = np.exp(1j * phases[i, j])
#         S_dot_r0 = (S / r0).real

#         a = 2 / temperature * S0_dot_r0 * S_dot_r0
#         prob = (a < 0) * (1 - np.exp(a))
#         if np.random.random() < prob:
#             S -= 2 * S_dot_r0 * r0
#             phases[i, j] = np.angle(S)
#             visited[i, j] = True

#             phases, visited = construct_wolff_cluster(
#                 i,
#                 j,
#                 r0,
#                 phases,
#                 neighbors,
#                 scales,
#                 visited,
#                 temperature,
#             )

#     return phases, visited


# @numba.njit(fastmath=True)
# def run_wolff_step(
#     phases: np.ndarray, neighbors: np.ndarray, scales: np.ndarray, temperature: float
# ) -> np.ndarray:
#     """Performs a single Wolff update by constructing and flipping a cluster."""

#     nrows, ncols = phases.shape
#     visited = np.zeros((nrows, ncols), dtype=numba.boolean)

#     # Pick initial site at random
#     row = np.random.randint(0, nrows)
#     col = np.random.randint(0, ncols)
#     S = np.exp(1j * phases[row, col])
#     r0 = np.exp(1j * 2 * np.pi * np.random.random())

#     # Reflect spin at initial site aboout the plane perpendicular to r0
#     S_dot_r0 = (S / r0).real
#     S -= 2 * S_dot_r0 * r0
#     phases[row, col] = np.angle(S)
#     visited[row, col] = True

#     # Build cluster starting from initial site
#     phases, visited = construct_wolff_cluster(
#         row, col, r0, phases, neighbors, scales, visited, temperature
#     )
#     return phases


def run_model(
    temperature: float,
    *,
    nrows: int,
    ncols: int,
    loop_center: Tuple[float, float, float],
    loop_radius: float,
    loop_current: float,
    J: float = 1.0,
    lattice_constant: float = 1.0,
    pickup_loop_center: Optional[Tuple[float, float, float]] = None,
    pickup_loop_radius: float = 3.0,
    hot_start: bool = False,
    metropolis_steps_per_pass: Optional[int] = None,
    thermalize_passes: int = 1_000,
    measure_passes: int = 10_000,
    periodic: bool = False,
    rng_seed: Optional[int] = None,
    progress_bar: bool = True,
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

    island_centers = make_island_centers(nrows, ncols, lattice_constant)
    A = get_vector_potential(loop_center, loop_radius, loop_current, island_centers)
    A *= lattice_constant
    neighbors, scales = get_all_neighbors_scales(nrows, ncols, periodic=periodic)
    Phi_0 = ureg("Phi_0").to("T * m**2").magnitude

    if pickup_loop_center is not None:
        x0, y0, z0 = pickup_loop_center
        pl = sc.Polygon(points=sc.geometry.circle(pickup_loop_radius, center=(x0, y0)))
        pl_mesh = pl.make_mesh(max_edge_length=pickup_loop_radius / 15)
        pl_points = 1e-6 * np.append(
            pl_mesh.sites, z0 * np.ones((len(pl_mesh.sites), 1)), axis=1
        )
        pl_areas = (1e-6) ** 2 * pl_mesh.vertex_areas

        bond_centers = island_centers[:-1, :-1] + lattice_constant / 2
        bond_centers = bond_centers.reshape((-1, 2))
        bond_centers = 1e-6 * np.append(
            bond_centers, np.zeros((len(bond_centers), 1)), axis=1
        )

    # Thermalize
    for _ in tqdm(
        range(thermalize_passes), desc="Thermalizing", disable=(not progress_bar)
    ):
        phases = run_n_metropolis_steps(
            metropolis_steps_per_pass,
            phases,
            A,
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
    _squid_susc = np.zeros(measure_passes)

    for i in tqdm(range(measure_passes), desc="Measuring", disable=(not progress_bar)):
        phases = run_n_metropolis_steps(
            metropolis_steps_per_pass,
            phases,
            A,
            neighbors,
            scales,
            temperature,
            J=J,
        )

        _energy[i] = calculate_energy(phases, A, neighbors, scales)
        M = calculate_magnetization(phases)
        _magnetization[i] = M
        _magnetization2[i] = M**2
        if periodic:
            ex, sx, ey, sy = calculate_helicity_periodic(phases, J=J)
        else:
            ex, sx, ey, sy = calculate_helicity_open(phases, J=J)
        _helicity_x_e[i] = ex
        _helicity_x_s2[i] = sx**2
        _helicity_y_e[i] = ey
        _helicity_y_s2[i] = sy**2

        if pickup_loop_center is not None:
            Ix, Iy = get_currents(phases, A)
            currents = J * np.array([Ix, Iy]).transpose(1, 2, 0)
            currents = currents.reshape((-1, 2)) * (1e-6 * lattice_constant)
            Bz = biot_savart_from_lattice(pl_points, bond_centers, currents)
            _squid_susc[i] = np.sum(Bz * pl_areas) / Phi_0 / loop_current

    energy = ufloat(np.mean(_energy), np.std(_energy))
    magnetization = ufloat(np.mean(_magnetization), np.std(_magnetization))
    magnetization2 = ufloat(np.mean(_magnetization2), np.std(_magnetization2))
    susceptibility = (magnetization2 - magnetization**2) * phases.size / temperature
    squid_susc = ufloat(np.mean(_squid_susc), np.std(_squid_susc))

    helicity_x_e = ufloat(np.mean(_helicity_x_e), np.std(_helicity_x_e))
    helicity_x_s2 = ufloat(np.mean(_helicity_x_s2), np.std(_helicity_x_s2))
    helicity_y_e = ufloat(np.mean(_helicity_y_e), np.std(_helicity_y_e))
    helicity_y_s2 = ufloat(np.mean(_helicity_y_s2), np.std(_helicity_y_s2))

    if periodic:
        Nx = Ny = phases.size
    else:
        Nx = nrows * (ncols - 1)
        Ny = (nrows - 1) * ncols

    helicity_x = (1 / Nx) * (helicity_x_e - helicity_x_s2 / temperature)
    helicity_y = (1 / Ny) * (helicity_y_e - helicity_y_s2 / temperature)

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
        squid_susceptibility=squid_susc,
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
    loop_center: Tuple[float, float, float],
    loop_radius: float,
    loop_current: float,
    J: float = 1.0,
    lattice_constant: float = 1.0,
    pickup_loop_center: Optional[Tuple[float, float, float]] = None,
    pickup_loop_radius: float = 3.0,
    hot_start: bool = False,
    metropolis_steps_per_pass: Optional[int] = None,
    thermalize_passes: int = 1_000,
    measure_passes: int = 10_000,
    periodic: bool = False,
    progress_bar: bool = True,
    rng_seed: Optional[int] = None,
) -> List[XYResult]:

    func = partial(
        run_model,
        J=J,
        nrows=nrows,
        ncols=ncols,
        loop_center=loop_center,
        loop_radius=loop_radius,
        loop_current=loop_current,
        lattice_constant=lattice_constant,
        pickup_loop_center=pickup_loop_center,
        pickup_loop_radius=pickup_loop_radius,
        hot_start=hot_start,
        metropolis_steps_per_pass=metropolis_steps_per_pass,
        thermalize_passes=thermalize_passes,
        measure_passes=measure_passes,
        periodic=periodic,
        rng_seed=rng_seed,
        progress_bar=False,
    )

    if num_cpus > 1:
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.map(func, temperatures)
    else:
        results = []
        for temp in tqdm(temperatures, desc="Temperatures", disable=(not progress_bar)):
            results.append(func(temp))

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--output-path", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--nrows", type=int)
    parser.add_argument("--ncols", type=int)
    parser.add_argument("--loop-center", type=float, nargs=3)
    parser.add_argument("--loop-radius", type=float)
    parser.add_argument("--loop-current", type=float, default=100)
    parser.add_argument("--lattice-constant", type=float)
    parser.add_argument("--pickup-loop-center", type=float, nargs=3, default=None)
    parser.add_argument("--pickup-loop-radius", type=float, default=3.0)
    parser.add_argument("--metropolis-steps-per-pass", type=int, default=None)
    parser.add_argument("--thermalize-passes", type=int)
    parser.add_argument("--measure-passes", type=int)
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument("--hot-start", action="store_true")
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")

    args = parser.parse_args()
    kwargs = vars(args)
    output_path = kwargs.pop("output_path")
    result = run_model(**kwargs)

    with open(output_path, "x") as f:
        json.dump(result.to_json(), f, indent=4)


if __name__ == "__main__":
    main()
