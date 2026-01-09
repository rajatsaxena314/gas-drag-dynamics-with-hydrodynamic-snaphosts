import numpy as np
from tqdm import tqdm

def exponential_mid_point(x0, y0, z0, vx0, vy0, vz0, G,
                                   n_steps, dt, compute_values, helper, mass_enclosed,
                                   t_unit=1, v_unit=1, r_unit=1):
    """
    N-body integrator with exponential implicit velocity update for gas drag.

    Parameters
    ----------
    x0, y0, z0 : float
        Initial particle positions.
    vx0, vy0, vz0 : float
        Initial particle velocities.
    n_steps : int
        Number of integration steps.
    dt : float
        Timestep.
    compute_values : function
        Function returning gas velocity and stopping time: 
        v_g_az, v_g_r, v_g_ang, t_stop = compute_values(state)
    helper : module
        Module with spherical_velocity_to_cartesian function.
    t_unit, v_unit : float
        Normalization units for time and velocity.

    Returns
    -------
    x, y, z, vx, vy, vz, time : np.ndarray
        Arrays of particle positions, velocities, and time.
    """

    # Initialize arrays
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    vx = np.zeros(n_steps)
    vy = np.zeros(n_steps)
    vz = np.zeros(n_steps)
    vgx = np.zeros(n_steps)
    vgy = np.zeros(n_steps)
    vgz = np.zeros(n_steps)
    time = np.zeros(n_steps)
    # Set initial conditions
    x[0], y[0], z[0] = x0, y0, z0
    vx[0], vy[0], vz[0] = vx0, vy0, vz0
    int1, int2, int3, _ = compute_values(np.array([x0, y0, z0, vx0, vy0, vz0, time[0]]))
    vgx[0], vgy[0], vgz[0] = helper.spherical_velocity_to_cartesian(x[0]*r_unit, y[0]*r_unit, z[0]*r_unit, int1, int2, int3)
    vgx[0], vgy[0], vgz[0] = vgx[0]/v_unit, vgy[0]/v_unit, vgz[0]/v_unit 
    for i in tqdm(range(1, n_steps)):
        # Time keeping
        time[i] = i * dt
        t_n_half = time[i] + dt

        # Half-step position (kick)
        x_half = x[i-1] + cnst * vx[i-1]
        y_half = y[i-1] + cnst * vy[i-1]
        z_half = z[i-1] + cnst * vz[i-1]

        # System state for gas computation
        system_state = np.array([x_half, y_half, z_half,
                              vx[i-1], vy[i-1], vz[i-1],
                              t_n_half])

        # Compute gas velocity and stopping time
        v_g_az, v_g_r, v_g_ang, t_stop = compute_values(system_state)
        M_enc = mass_enclosed(system_state)
        #t_stop/= t_unit

        # Convert gas velocity to Cartesian coordinates
        v_g_x, v_g_y, v_g_z = helper.spherical_velocity_to_cartesian(
            x_half*r_unit, y_half*r_unit, z_half*r_unit, v_g_r, v_g_ang, v_g_az
        )
        vgx[i], vgy[i], vgz[i] = v_g_x/v_unit, v_g_y/v_unit, v_g_z/v_unit

        # Exponential implicit velocity update
        b = dt / t_stop
        h = t_stop * (1 - np.exp(-dt / t_stop))

        # Gravitational acceleration + gas drag
        radius = np.sqrt(x_half**2 + y_half**2 + z_half**2)
        a_x_new = -(G*M_enc*x_half)/radius**3
        a_y_new = -(G*M_enc*y_half)/radius**3
        a_z_new = -(G*M_enc*z_half)/radius**3

        vx[i] = vgx[i] + (vx[i - 1] - vgx[i - 1]) * np.exp(-dt / t_stop) + h * a_x_new
        vy[i] = vgy[i] + (vy[i - 1] - vgy[i - 1]) * np.exp(-dt / t_stop) + h * a_y_new
        vz[i] = vgz[i] + (vz[i - 1] - vgz[i - 1]) * np.exp(-dt / t_stop) + h * a_z_new


        # Full-step position (kick)
        x[i] = x_half + dt * vx[i]
        y[i] = y_half + dt * vy[i]
        z[i] = z_half + dt * vz[i]

    return x, y, z, vx, vy, vz, time