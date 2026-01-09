import numpy as np
from types import SimpleNamespace

def generate_grids(dataset, sorted_keys):
    """
    Generate radial and angular grids (and their interfaces) from a dataset.
    This function extracts the theta (angular) and r (radial) grids from the provided dataset,
    checks if the grids are linear or logarithmic, and computes the corresponding interface grids.
    It also computes the Cartesian coordinates for both cell centers and interfaces.

    Parameters
    ----------
    dataset : dict-like or h5py-like object
        The dataset containing the grid data. It must support key-based access to arrays.
    sorted_keys : list of str
        List of keys (strings) used to access the relevant data in the dataset. The first key is used.

    Returns
    -------
    SimpleNamespace
        An object with the following attributes:
            - theta: 1D array of angular grid centers.
            - r: 1D array of radial grid centers.
            - theta_i: 1D array of angular grid interfaces.
            - r_i: 1D array of radial grid interfaces.
            - x: 2D array of x-coordinates of cell centers.
            - y: 2D array of y-coordinates of cell centers.
            - x_i: 2D array of x-coordinates of cell interfaces.
            - y_i: 2D array of y-coordinates of cell interfaces.
    Raises
    ------
    NotImplementedError
        If the theta or r grids are not linear or logarithmic (for r).
    """

    theta = dataset[f'{sorted_keys[0]}/theta'][0, :]
    r = dataset[f'{sorted_keys[0]}/radius'][:, 0]

    A = r[1:] / r[:-1]
    dtheta = np.diff(theta)
    dr = np.diff(r)

    if np.allclose(dtheta, dtheta[0]):
        print('linear theta grid')
        theta_i = np.hstack((theta - 0.5 * dtheta.mean(),
                            theta[-1] + 0.5 * dtheta.mean()))

    else:
        raise NotImplementedError('non-linear theta not implemented')

    if np.allclose(dr, dr[0]):
        print('linear radius grid')
        r_i = np.hstack((r - 0.5 * dr.mean(), r[-1] + 0.5 * dr.mean()))
    elif np.allclose(A, A[0]):
        print('log radius grid')
        A = A.mean()
        r_i = np.hstack((2 / (A + 1) * r[0], 2 * A / (A + 1) * r))
    else:
        raise NotImplementedError('non-linear r not implemented')

    x = r[:, None] * np.sin(theta[None, :])
    y = r[:, None] * np.cos(theta[None, :])
    x_i = r_i[:, None] * np.sin(theta_i[None, :])
    y_i = r_i[:, None] * np.cos(theta_i[None, :])

    return SimpleNamespace(
        theta=theta,
        r=r,
        r_i=r_i,
        theta_i=theta_i,
        x_i=x_i,
        y_i=y_i,
        x=x,
        y=y,
    )

def position_mapper(x,y,z):
    return np.abs(x), np.abs(y), np.abs(z)

def axisymmetric_spherical_to_cartesian_2p5D(r, theta, v_r, v_theta, v_phi):
    """
    Convert spherical coordinates and velocities to Cartesian form in 2.5D simulations.

    Assumes azimuthal angle φ = 0 (axisymmetry), so particles lie in the xz-plane.

    Parameters:
    - r: radial position (shape N,)
    - theta: polar angle from z-axis (shape N,)
    - v_r: radial velocity (shape N,)
    - v_theta: polar angular velocity (shape N,)
    - v_phi: azimuthal velocity (shape N,)

    Returns:
    - positions: (N, 3) array of [x, y, z]
    - velocities: (N, 3) array of [vx, vy, vz]
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Position in xz-plane
    x = r * sin_theta #Normally this would r*cos(theta) but for PLUTO it's opposite
    y = np.zeros_like(r)
    z = r * cos_theta #Normally this would r*sin(theta) but for PLUTO it's opposite

    # Velocity components
    vx = v_r * sin_theta + v_theta * cos_theta
    vy = v_phi
    vz = v_r * cos_theta - v_theta * sin_theta

    positions = np.stack((x, y, z), axis=-1)
    velocities = np.stack((vx, vy, vz), axis=-1)

    return positions, velocities

def spherical_velocity_to_cartesian(x, y, z, v_r, v_theta, v_phi):
    """
    Convert spherical velocity components (radial, polar, azimuthal)
    at a given Cartesian position to Cartesian velocity components.

    Parameters:
        x, y, z : float
            Cartesian position
        v_r : float
            Radial velocity component
        v_theta : float
            Polar (theta) velocity component
        v_phi : float
            Azimuthal (phi) velocity component

    Returns:
        vx, vy, vz : float
            Cartesian velocity components
    """
    r = np.sqrt(x**2 + y**2 + z**2) #AU
    theta = np.arccos(z / r)          # polar angle from z-axis
    phi = np.arctan2(y, x)            # azimuthal angle in x-y plane

    # Convert spherical velocities to Cartesian velocities
    vx = v_r * np.sin(theta) * np.cos(phi) + v_theta * np.cos(theta) * np.cos(phi) - v_phi * np.sin(phi)
    vy = v_r * np.sin(theta) * np.sin(phi) + v_theta * np.cos(theta) * np.sin(phi) + v_phi * np.cos(phi)
    vz = v_r * np.cos(theta) - v_theta * np.sin(theta)
    
    return vx, vy, vz