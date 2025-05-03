import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Van der Pol Oscillator with forcing
def van_der_pol(t,x z, mu, omega, F):
    x, v = z
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x + F * np.cos(omega * t)
    return [dxdt, dvdt]

# Function to generate fixed points using numerical integration
def fixed_point_bifurcation(mu, omega, F_values, t_max=1000, dt=0.01, initial_conditions=[0.1, 0]):
    x_fixed_points = []

    for F in F_values:
        t_eval = np.arange(0, t_max, dt)
        
        # Solve the Van der Pol oscillator
        sol = solve_ivp(van_der_pol, [0, t_max], initial_conditions, args=(mu, omega, F), t_eval=t_eval)
        
        # Extract the final values of x (after transients)
        x_final = sol.y[0][-100:]  # Last 100 points to avoid transients
        x_fixed_points.append(np.mean(x_final))  # Use the mean of the last 100 points as the fixed point

    return np.array(x_fixed_points)

# Parameters for the Van der Pol Oscillator
mu = 1.0  # Nonlinearity parameter
omega = 3.0  # Frequency of forcing
F_values = np.linspace(0.01, 4, 1000)  # Forcing amplitude range

# Generate the fixed point bifurcation diagram
x_fixed_values = fixed_point_bifurcation(mu, omega, F_values)

# Plot the bifurcation diagram for fixed points
plt.figure(figsize=(10, 6))
plt.plot(F_values, x_fixed_values, 'k.', markersize=1)
plt.title(f'Fixed Point Bifurcation Diagram for μ={mu}, ω={omega}')
plt.xlabel('F (Forcing Amplitude)')
plt.ylabel('Fixed Points (x)')
plt.show()

# Function to generate Time Series and Phase Portrait for a given F
def plot_time_series_and_phase_portrait(mu, omega, F, t_max=1000, dt=0.01, initial_conditions=[0.1, 0]):
    t_eval = np.arange(0, t_max, dt)
    
    # Solve the Van der Pol oscillator
    sol = solve_ivp(van_der_pol, [0, t_max], initial_conditions, args=(mu, omega, F), t_eval=t_eval)
    
    # Extract solution data
    x = sol.y[0]
    v = sol.y[1]
    
    # Plot Time Series (x vs time)
    plt.figure(figsize=(12, 5))
    
    # Time Series graph on the right (x vs time for 400 < t < 500)
    plt.subplot(1, 2, 2)
    time_range = (t_eval >= 400) & (t_eval <= 500)
    plt.plot(t_eval[time_range], x[time_range])
    plt.title(f'Time Series for F={F}, μ={mu}, ω={omega}')
    plt.xlabel('Time')
    plt.ylabel('x')
    
    # Phase Portrait (x vs v)
    plt.subplot(1, 2, 1)
    plt.plot(x, v)
    plt.title(f'Phase Portrait for F={F}, μ={mu}, ω={omega}')
    plt.xlabel('x')
    plt.ylabel('v')
    
    plt.tight_layout()
    plt.show()

# Five specific F values to plot time series and phase portraits
F_values_to_plot = [0.05]

# Generate Time Series and Phase Portrait for each selected F value
for F in F_values_to_plot:
    plot_time_series_and_phase_portrait(mu, omega, F)
