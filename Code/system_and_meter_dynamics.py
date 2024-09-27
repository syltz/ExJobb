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
def plot_joint_probabilities(sam):
    """Returns a plot of the joint probabilities as a function of time of the system being in state 0 or 1 and the meter being in state n.

    Args:
        sam (SystemAndMeter): The system and meter object to plot the joint probabilities for.
    
    Returns:
        ax (matplotlib.axes.Axes): The axes object of the plot.
    """
    p0, p1 = sam.joint_prob_evol()
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(p0))*(sam.get_time()/len(p0)), p0, label='p0', color='blue')
    ax.plot(np.arange(0, len(p1))*(sam.get_time()/len(p1)), p1, label='p1', color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.legend()
    plt.close(fig)
    return ax

temp_meter = 10
temp_system = 1
omega_meter = 2*np.pi
time = 10/omega_meter
coupling = 1/np.sqrt(hbar*omega_meter/2)
init_system_state = (1/2, 1/2)
measurement_state = 1
total_levels = 10
sam = SystemAndMeter(temp_meter=temp_meter, temp_system=temp_system, omega_meter=omega_meter,\
                        coupling=coupling, time=time, init_system_state=init_system_state,\
                        measurement_state=measurement_state, total_levels=total_levels)

times = np.array([time, time/2, time/10])
fig, axs = plt.subplots(len(times), 1, figsize=(10, 8))
plt.title('Conditional Probabilities as a function of time measuring in n=1')
for i,t in enumerate(times):
    sam.set_time(t)
    ax = plot_conditional_probabilities(sam)
    ax2 = plot_joint_probabilities(sam)
    axs[i].plot(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), label=r'$P_0(t|n)$', color='blue')
    axs[i].plot(ax.lines[1].get_xdata(), ax.lines[1].get_ydata(), label=r'$P_1(t|n)$', color='red')
    #axs[i].plot(ax.lines[2].get_xdata(), ax.lines[2].get_ydata(), label=r'$P_0(t|n)+P_1(1|n)$', color='black')
    #axs[i].plot(ax2.lines[0].get_xdata(), ax2.lines[0].get_ydata(), label=r'$P_0(n,t)$', color='blue', linestyle='--')
    #axs[i].plot(ax2.lines[1].get_xdata(), ax2.lines[1].get_ydata(), label=r'$P_1(n,t)$', color='red', linestyle='--')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Probability')
    axs[i].legend()
plt.tight_layout()
plt.savefig(f'../images/conditional_probabilities_Tm_{temp_meter}.png')

fig, ax = plt.subplots()
time = 0.9
sam.set_time(time)
sam.set_temp_meter(10)
sam.full_update()

meas_levels = np.arange(0, 10)#sam.get_total_levels()+1)
joint_probs = np.zeros_like(meas_levels, dtype=np.float64)
joint_probs_0 = np.zeros_like(meas_levels, dtype=np.float64)
cond_probs = np.zeros_like(meas_levels, dtype=np.float64)
for i in range(len(meas_levels)):
    sam.set_n(i) # Set the meter level to i
    p0_n, p1_n = sam.joint_probability(sam.get_n(), sam.get_time())
    joint_probs[i] = p1_n
    joint_probs_0[i] = p0_n
    cond_probs[i] = p1_n/(p0_n+p1_n)
#x = np.arange(0, 10)
#y = np.exp(-(sam.get_gamma()+0.5)*x/100)
#y = sam.get_tls_state()[0]*y/np.sum(y)
#ax.plot(x, y, label='TLS State 0', color='black')
ax.scatter(meas_levels, joint_probs, s=10, color='red', label='P_1(n,t)')
ax.scatter(meas_levels, cond_probs, s=10, color='blue', label='P(1|n,t)')
ax.scatter(meas_levels, joint_probs_0, s=10, color='green', label='P_0(n,t)')
ax.set_xlabel('Meter Level')
ax.set_ylabel('Probability')
ax.legend()
ax.set_title(f'Prob of i given meter in n time t={time:.3f}')
plt.savefig(f'../images/joint_probabilities_Tm_{sam.get_temp_meter()}_tm_{sam.get_time()}.png')

# Plot conditional entropy
fig, axs = plt.subplots(3,1)
times = np.array([0.5, 0.75, 1.0])
for i,t in enumerate(zip(times,axs)):
    sam.set_time(t[0])
    cond_entropy = np.zeros(sam.get_total_levels())
    for i in range(sam.get_total_levels()):
        cond_entropy[i] = sam.conditional_entropy(n=i)
    t[1].hlines(cond_entropy[0], 0, sam.get_total_levels(), color='black', linestyle='--', label='Meter Level 0')
    t[1].scatter(np.arange(0, sam.get_total_levels()), cond_entropy, label=f'$S_n(t={t[0]})$')
    t[1].set_xlabel('Meter Level')
    t[1].set_ylabel('Conditional Entropy')
    t[1].set_title(f'Conditional Entropy as a function of meter level at time t={sam.get_time()}')
plt.tight_layout()
plt.savefig(f'../images/conditional_entropy_Tm_{sam.get_temp_meter()}.png')

# Plot entropy and conditional entropy as a function of time 
fig, ax = plt.subplots()
times = np.linspace(0.1, 2, 100)
entropy = np.zeros_like(times)
cond_entropy = np.zeros_like(times)
sam.set_n(0)
for i,t in enumerate(times):
    sam.set_time(t)
    entropy[i] = sam.entropy()
    cond_entropy[i] = sam.conditional_entropy()
ax.plot(times, entropy, color='blue', label='Entropy')
ax.plot(times, cond_entropy, color='red', label='Conditional Entropy')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Entropy')
ax.set_title('Entropy as a function of time')
plt.savefig(f'../images/entropy_Tm_{sam.get_temp_meter()}.png')

# Plot mutual information as a function of time
mut_info = np.zeros_like(times)
fig, ax = plt.subplots()
for i, t in enumerate(times):
    #sam.set_time(t)
    mut_info[i] = sam.mutual_information(time = t)
ax.plot(times, mut_info, color='blue', label='Mutual Information')
ax.set_xlabel('Time')
ax.set_ylabel('Mutual Information')
ax.set_title('Mutual information between system and meter as a function of time')
plt.savefig(f'../images/mutual_information_Tm_{sam.get_temp_meter()}.png')

# Plot observer information at time t=0.9 as a function of meter level
fig, ax = plt.subplots()
sam.set_time(0.9)
for n in range(sam.get_total_levels()):
    sam.set_n(n)
    obs_info = sam.observer_information()
    ax.scatter(n, obs_info, color='blue', label='Observer Information')
plt.savefig(f'../images/observer_information_Tm_{sam.get_temp_meter()}.png')