import numpy as np
from tqdm import tqdm

def epsilon_implicit(x0, y0, z0, vx0, vy0, vz0, n_steps, dt, 
                     G, compute_values, mass_enclosed, helper, t_unit=1.0, v_unit=1.0, r_unit=1.0):
    """
    N-body integrator with epsilon-implicit velocity update and gas drag.
    
    Parameters
    ----------
    x0, y0, z0 : float
        Initial positions of the particle.
    vx0, vy0, vz0 : float
        Initial velocities of the particle.
    n_steps : int
        Number of integration steps.
    dt : float
        Timestep.
    G : float
        Gravitational constant.
    M_central : float
        Mass of the central body.
    compute_values : function
        Function to compute gas velocities and stopping time:
        v_g_az, v_g_r, v_g_ang, t_stop = compute_values(state)
    helper : module
        Module with spherical_velocity_to_cartesian function.
    t_unit, v_unit : float
        Normalization units for time and velocity.

    Returns
    -------
    x, y, z, vx, vy, vz, time : np.ndarray
        Arrays containing the particle trajectory and velocities.
    """
    
    # Initialize arrays
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    vx = np.zeros(n_steps)
    vy = np.zeros(n_steps)
    vz = np.zeros(n_steps)
    time = np.zeros(n_steps)
    
    # Set initial conditions
    x[0], y[0], z[0] = x0, y0, z0
    vx[0], vy[0], vz[0] = vx0, vy0, vz0
    
    for i in tqdm(range(1, n_steps)):
        # Time keeping
        time[i] = i * dt
        cnst = dt/2
        t_n_half = time[i] + cnst
        
        # Half-step position (kick)
        x_half = x[i-1] + cnst * vx[i-1]
        y_half = y[i-1] + cnst * vy[i-1]
        z_half = z[i-1] + cnst * vz[i-1]
        
        # System state for computing gas drag
        sys_state = np.array([x_half, y_half, z_half, vx[i-1], vy[i-1], vz[i-1], t_n_half])
        
        # Compute gas velocity and stopping time
        v_g_az, v_g_r, v_g_ang, t_stop = compute_values(sys_state)
        M_enc = mass_enclosed(sys_state)
        #t_stop /= t_stop
        
        # Convert gas velocity to Cartesian
        v_g_x, v_g_y, v_g_z = helper.spherical_velocity_to_cartesian(
            x_half*r_unit, y_half*r_unit, z_half*r_unit, v_g_r, v_g_ang, v_g_az
        )
        v_g_x, v_g_y, v_g_z = v_g_x/v_unit, v_g_y/v_unit, v_g_z/v_unit
        
        # Implicit velocity update coefficients
        b = dt/t_stop
        e = 0.5
        c = (1)/(1-(b*e))
        
        # Gravitational acceleration + gas drag
        radius = np.sqrt(x_half**2 + y_half**2 + z_half**2)
        a_x_new = -(G*M_enc*x_half)/radius**3 - (vx[i-1] - v_g_x)/t_stop
        a_y_new = -(G*M_enc*y_half)/radius**3 - (vy[i-1] - v_g_y)/t_stop
        a_z_new = -(G*M_enc*z_half)/radius**3 - (vz[i-1] - v_g_z)/t_stop
        
        # Update velocities
        vx[i] = vx[i-1] + dt * c * a_x_new
        vy[i] = vy[i-1] + dt * c * a_y_new
        vz[i] = vz[i-1] + dt * c * a_z_new

        # Full-step position (kick)
        x[i] = x_half + cnst * vx[i]
        y[i] = y_half + cnst * vy[i]
        z[i] = z_half + cnst * vz[i]
        
    return x, y, z, vx, vy, vz, time