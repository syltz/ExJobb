import numpy as np
import matplotlib.pyplot as plt
from SystemAndMeter import SystemAndMeter

# Throughout this code, the meter is assumed to be a harmonic oscillator
# and the system is assumed to be a two-level system.
# I also use mass = 1, hbar = 1, and kB = 1 for simplicity. I do however keep them
# in the equations for clarity and so that they can be easily changed if needed.
# It's mostly to avoid any potential numerical issues that may arise from the
# small values of hbar and kB.
hbar = 1.0 # sp.constants.hbar
kB = 1.0 # sp.constants.k

def plot_conditional_probabilities(sam):
    """Returns a plot of the conditional probabilities as a function of time of the system being in state 0 or 1 given the meter is in state n.

    Args:
        sam (SystemAndMeter): The system and meter object to plot the conditional probabilities for.
    
    Returns:
        ax (matplotlib.axes.Axes): The axes object of the plot.
    """
    p0, p1 = sam.prob_evol()
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(p0))*(sam.get_time()/len(p0)), p0, label='p0', color='blue')
    ax.plot(np.arange(0, len(p1))*(sam.get_time()/len(p1)), p1, label='p1', color='red')
    ax.plot(np.arange(0, len(p1))*(sam.get_time()/len(p1)),p0+p1, label='p0+p1', color='black')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.legend()
    plt.close(fig)
    return  ax

temp_meter = 1
temp_system = 1
omega_meter = 2*np.pi
time = 10/omega_meter
coupling = 1
init_system_state = (1/2, 1/2)
measurement_state = 1
sam = SystemAndMeter(temp_meter=temp_meter, temp_system=temp_system, omega_meter=omega_meter,\
                        coupling=coupling, time=time, init_system_state=init_system_state,\
                        measurement_state=measurement_state)

times = np.array([time, time/2, time/10])
fig, axs = plt.subplots(len(times), 1, figsize=(10, 8))
for i,t in enumerate(times):
    sam.set_time(t)
    sam.full_update()
    ax = plot_conditional_probabilities(sam)
    axs[i].plot(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), label='p0', color='blue')
    axs[i].plot(ax.lines[1].get_xdata(), ax.lines[1].get_ydata(), label='p1', color='red')
    axs[i].plot(ax.lines[2].get_xdata(), ax.lines[2].get_ydata(), label='p0+p1', color='black')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Probability')
    axs[i].legend()
plt.tight_layout()
plt.show()
