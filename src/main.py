import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from solve_1d_heat_pipe import solve_1d_heat_pipe

if __name__ == "__main__":
    nx = 100 # Number of nodes: 
    alpha = 0.001 # Thermal diffusivity
    dx = 1.0/(nx-1) 
    dt = 0.02 # Smaller time step for stability
    steps = 100000
    
    # Heat flux parameters (adjust values as needed)
    heat_flux = 0.2       # Heat flux in normalized units
    cooling_coeff = 0.1  # Cooling coefficient
    
    # Initial condition (starting from zero temperature)
    T = np.zeros(nx)
    
    # Solve
    T_history = solve_1d_heat_pipe(T, alpha, dx, dt, steps, heat_flux, cooling_coeff)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [])
    
    # Add shaded regions to show heating and cooling zones
    ax.axvspan(0, nx//4, alpha=0.2, color='red', label='Heat Input')
    ax.axvspan(3*nx//4, nx, alpha=0.2, color='blue', label='Cooling Zone')
    
    # Add lines to show initial value of temperature
    max_temp = np.max(T_history)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlim(0, nx-1)
    ax.set_ylim(0, np.max(T_history) * 1.1) # Dynamic y-limit based on max temperature
    ax.set_xlabel('Position')
    ax.set_ylabel('Temperature')
    ax.set_title('Heat Pipe Simulation')
    ax.legend(loc='upper right')
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        # Skip frames to make animation smoother and faster
        frame = i * 20  # Show every 20th frame
        if frame < len(T_history):
            x = np.arange(0, nx)
            y = T_history[frame]
            line.set_data(x, y)
            # Update title with time information
            ax.set_title(f'Heat Pipe Simulation - Time Step: {frame}')
        return line,
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                         frames=steps//20, interval=30, blit=True)
    
    plt.show()
    
    # Uncomment the line below to save the animation
    # anim.save('heat_pipe_simulation.mp4', writer='ffmpeg', fps=30)

