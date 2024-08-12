#!/usr/bin/env python3
"""Partial Differential Equations to use as forward models."""

from typing import Optional

import geostat
import numpy as np
import numpy.typing as npt
from definition import (
    dx,
    k_end,
    k_start,
    nx,
    obs_coordinates,
    obs_times,
    u_3d,
)


def heat_equation(
    u: npt.NDArray[np.float64],
    cond: npt.NDArray[np.float64],
    dx: int,
    dt: float,
    k_start: int,
    k_end: int,
    rng: np.random.Generator,
    scale: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    """2D heat equation that suppoheat_erts field of heat coefficients.

    Based on:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """
    _u = u.copy()
    nx = u.shape[1]  # number of grid cells
    assert cond.shape == (nx, nx)

    gamma = (cond * dt) / (dx**2)
    plate_length = u.shape[1]
    for k in range(k_start, k_end - 1, 1):
        for i in range(1, plate_length - 1, dx):
            for j in range(1, plate_length - 1, dx):
                noise = rng.normal(scale=scale) if scale is not None else 0
                _u[k + 1, i, j] = (
                    gamma[i, j]
                    * (
                        _u[k][i + 1][j]
                        + _u[k][i - 1][j]
                        + _u[k][i][j + 1]
                        + _u[k][i][j - 1]
                        - 4 * _u[k][i][j]
                    )
                    + _u[k][i][j]
                    + noise
                )

    return _u


def heat_equation_3d(
    u: npt.NDArray[np.float64],
    alpha: npt.NDArray[np.float64],
    dx: int,
    dt: float,
    k_start: int,
    k_end: int,
    rng: np.random.Generator,
    scale: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    """3D heat equation that supports field of heat coefficients."""
    _u = u.copy()
    nz, ny, nx = u.shape[1:]  # number of grid cells in each dimension
    print(alpha.shape)
    assert alpha.shape == (nx, ny, nz)
    gamma = (alpha * dt) / (dx**2)

    for k in range(k_start, k_end - 1, 1):
        if nz == 1:  # 2D case embedded in 3D
            for j in range(1, ny - 1, dx):
                for l in range(1, nx - 1, dx):
                    if scale is not None:
                        noise = rng.normal(scale=scale)
                    else:
                        noise = 0
                    _u[k + 1, 0, j, l] = (
                        gamma[l, j, 0]  # Changed from gamma[0, j, l]
                        * (
                            _u[k][0][j + 1][l]
                            + _u[k][0][j - 1][l]
                            + _u[k][0][j][l + 1]
                            + _u[k][0][j][l - 1]
                            - 4 * _u[k][0][j][l]
                        )
                        + _u[k][0][j][l]
                        + noise
                    )
        else:  # Full 3D case
            for i in range(1, nz - 1, dx):
                for j in range(1, ny - 1, dx):
                    for l in range(1, nx - 1, dx):
                        if scale is not None:
                            noise = rng.normal(scale=scale)
                        else:
                            noise = 0
                        _u[k + 1, i, j, l] = (
                            gamma[l, j, i]  # Changed from gamma[i, j, l]
                            * (
                                _u[k][i + 1][j][l]
                                + _u[k][i - 1][j][l]
                                + _u[k][i][j + 1][l]
                                + _u[k][i][j - 1][l]
                                + _u[k][i][j][l + 1]
                                + _u[k][i][j][l - 1]
                                - 6 * _u[k][i][j][l]
                            )
                            + _u[k][i][j][l]
                            + noise
                        )

    return _u


def sample_prior_conductivity(nx, ny, nz=None, rng=None):
    """
    Sample prior conductivity for 2D or 3D cases.

    Parameters:
    - nx, ny: int, number of grid points in x and y directions
    - nz: int or None, number of grid points in z direction (None for 2D case)
    - rng: numpy.random.Generator object or None

    Returns:
    - numpy array of shape (nx, ny, nz) for 3D or (nx, ny) for 2D
    """
    if rng is None:
        rng = np.random.default_rng()

    if nz is None:
        # 2D case
        mesh = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    else:
        # 3D case
        mesh = np.meshgrid(
            np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.linspace(0, 1, nz)
        )

    fields = geostat.gaussian_fields(mesh, rng, N=1, r=0.8)

    if nz is None:
        # Reshape for 2D case
        return np.exp(fields.reshape(ny, nx)).T
    else:
        # Reshape for 3D case
        return np.exp(fields.reshape(nz, ny, nx)).transpose(2, 1, 0)


if __name__ == "__main__":
    # iens = int(sys.argv[1])
    iens = 42
    rng = np.random.default_rng(iens)
    cond = sample_prior_conductivity(nx=nx, ny=nx, nz=1, rng=rng)

    # Write the array to a GRDECL formatted file
    with open("cond.grdecl", "w", encoding="utf-8") as f:
        f.write("COND\n")  # Write the property name
        f.write("-- Conductivity data\n")  # Optional comment line

        # Write the data
        for row in cond[:, :, 0]:
            for value in row:
                f.write(f"{value:.6f} ")
            f.write("\n")

        f.write("/\n")  # End the data section with a slash

    # Calculate maximum `dt`.
    # If higher values are used, the numerical solution will become unstable.
    # Note that this could be avoided if we used an implicit solver.
    dt = dx**2 / (4 * max(np.max(cond), np.max(cond)))

    response = heat_equation_3d(u_3d, cond, dx, dt, k_start, k_end, rng)

    index = sorted((0, obs.x, obs.y) for obs in obs_coordinates)
    for time_step in obs_times:
        with open(f"gen_data_{time_step}.out", "w", encoding="utf-8") as f:
            for i in index:
                f.write(f"{response[time_step][i]}\n")
