from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import numpy as np

def interpolate_cartesian(x, y, r_in, theta_in, Q_in):
    """
    Interpolates values from a polar grid (r, theta) onto Cartesian coordinates (x, y).
    Parameters
    ----------
    x : array_like
        The x-coordinates in Cartesian space where interpolation is desired.
    y : array_like
        The y-coordinates in Cartesian space where interpolation is desired.
    r_in : array_like
        The 1D array of radial coordinates defining the input polar grid.
    theta_in : array_like
        The 1D array of angular coordinates (in radians) defining the input polar grid.
    Q_in : array_like
        The 2D array of values defined on the (r_in, theta_in) grid to be interpolated.
    Returns
    -------
    values_at_particles : ndarray
        Interpolated values at the specified (x, y) Cartesian coordinates.
    Notes
    -----
    - Uses `scipy.interpolate.RegularGridInterpolator` for interpolation.
    - Assumes `Q_in` is ordered as (r_in, theta_in).
    """

    interpolator = RegularGridInterpolator(
        (r_in, theta_in), Q_in, bounds_error=False, fill_value=0.0)

    r_ = np.log(np.sqrt(x**2 + y**2))
    theta_ = np.mod(np.arctan2(y, x), 2*np.pi)

    # Interpolate
    values_at_particles = interpolator((r_, theta_))

    return values_at_particles

def interpolate(data3D, time_array, target_time, kind='linear'):
    """
    Interpolates a 3D data array along the time axis to a specified target time.

    Parameters
    ----------
    data3D : np.ndarray
        A 3D NumPy array with shape (ntime, nr, ntheta).
    time_array : array-like
        1D array of time points corresponding to the first axis of `data3D`.
    target_time : float or array-like
        The time or times at which to interpolate the data.
    kind : str, optional
        Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.).
        Default is 'linear'.

    Returns
    -------
    interpolated_slice : np.ndarray
        Interpolated data at the specified `target_time`. The shape is (nr, ntheta) if `target_time` is a scalar,
        or (len(target_time), nr, ntheta) if `target_time` is array-like.

    Raises
    ------
    ValueError
        If `target_time` is outside the range of `time_array` (due to `bounds_error=True`).
    """

    # Interpolate along time axis
    interpolator = interp1d(time_array, data3D, axis=0,
                            kind=kind, bounds_error=False, fill_value='extrapolate')
    interpolated_slice = interpolator(target_time)  # shape (nr, ntheta)

    return interpolated_slice