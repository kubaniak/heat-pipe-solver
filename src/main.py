import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from solve_yoo import explicit_finite_difference_solver

if __name__ == "__main__":
    # Example parameters dictionary:
    params = {
        "L_t": 1.0,       # Total length [m]
        "L_e": 0.3,       # Evaporator length [m]
        "L_a": 0.4,       # Adiabatic length [m]
        "L_c": 0.3,       # Condenser length [m]
        "dx": 0.01,       # Spatial step [m]
        "dt": 1e-5,       # Time step [s]
        "t_end": 10.0,     # Total simulation time [s]
        "rho": 8000,      # Example density [kg/m^3]
        "c_p": 500,       # Example specific heat [J/(kg-K)]
        "k_w": 15,        # Wall thermal conductivity [W/(m-K)]
        "q_e": 1e4,       # Heat flux at evaporator [W/m^2]
        "sigma": 5.67e-8, # Stefan-Boltzmann constant [W/(m^2-K^4)]
        "epsilon": 0.65,  # Emissivity
        "T_inf": 300,     # Ambient temperature [K]
        
        # Vapor effective conductivity parameters:
        "P": 1e5,         # Saturated vapor pressure [Pa]
        "R_v": 0.01,      # Vapor core radius [m]
        "mu_v": 1e-5,     # Dynamic viscosity [Pa-s]
        "m_g": 4.65e-26,  # Molecular mass [kg]
        "k_B": 1.380649e-23, # Boltzmann constant [J/K]
        "R_g": 461.5,     # Specific gas constant [J/(kg-K)]
        "N_A": 6.022e23,  # Avogadro's number [1/mol]
        "h_lv": 2.26e6,   # Latent heat of vaporization [J/kg]
        "h_l": 2.0e6,     # Latent heat coefficient (evap/con) [J/kg]
        "h_v": 1.8e6,     # Vapor enthalpy (adiabatic) [J/kg]
        "M_g": 3.8e-26,   # Molar mass of vapor [kg/mol]
        
        # Sonic limit parameters:
        "A_c": np.pi * (0.01**2), # Cross-sectional area [m^2]
        "rho_v0": 1.0,    # Example vapor density at sonic condition [kg/m^3]
        "h_cl": 2.0e6,    # Characteristic latent heat [J/kg]
        "gamma": 1.4,     # Specific heat ratio (adiabatic index)
        "T_v0": 700.0,    # Reference vapor temperature for sonic limit [K]
        
        # Initial condition for temperature (set uniformly)
        "T_init": 100.0   # Initial temperature [K]
    }

    # Solve
    x, T_history = explicit_finite_difference_solver(params)
    
    # Calculate number of animation frames
    steps = len(T_history) // 20  # Divide by frame skip factor
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], label='Temperature')
        
    # Add line to show initial temperature
    plt.axhline(y=params["T_init"], color='gray', linestyle='--', alpha=0.5, label='Initial Temp')
    
    ax.set_xlim(0, x[-1])  # Set x limits based on the domain
    ax.set_ylim(0, np.max(T_history) * 1.1) # Dynamic y-limit based on max temperature
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Heat Pipe Simulation')
    ax.legend(loc='upper right')
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        # Skip frames to make animation smoother and faster
        frame = i * 20  # Show every 20th frame
        if frame < len(T_history):
            y = T_history[frame]
            line.set_data(x, y)
            # Update title with time information
            time_value = frame * params["dt"]
            ax.set_title(f'Heat Pipe Simulation - Time: {time_value:.3f} s')
        return line,
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                         frames=steps, interval=30, blit=True)
    
    plt.show()
    
    # Uncomment the line below to save the animation
    # anim.save('heat_pipe_simulation.mp4', writer='ffmpeg', fps=30)

