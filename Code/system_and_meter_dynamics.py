import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from SystemAndMeter import SystemAndMeter

# Throughout this code, the meter is assumed to be a harmonic oscillator
# and the system is assumed to be a two-level system.
# I also use mass = 1, hbar = 1, and kB = 1 for simplicity. I do however keep them
# in the equations for clarity and so that they can be easily changed if needed.
# It's mostly to avoid any potential numerical issues that may arise from the
# small values of hbar and kB.
hbar = 1.0 # sp.constants.hbar
kB = 1.0 # sp.constants.k

temp_meter = 1
temp_system = 1
omega_meter = 2*np.pi
time = 10/omega_meter
coupling = 5 #1/np.sqrt(hbar*omega_meter/2)
init_system_state = (1/2, 1/2)
#init_system_state = np.random.uniform(0, 1, 2)
measurement_state = 0
total_levels = 10
sam = SystemAndMeter(temp_meter=temp_meter, temp_system=temp_system, omega_meter=omega_meter,\
                        coupling=coupling, time=time, init_system_state=init_system_state,\
                        measurement_state=measurement_state, total_levels=total_levels)
sam.set_temp_meter(50)
sam.full_update()
print(sam.get_total_levels())
times = np.array([time, time/2, time/10])
fig, axs = plt.subplots(len(times), 1, figsize=(10, 8))
plt.title(f'Conditional Probabilities as a function of time measuring in state {sam.get_n()}')
for i,t in enumerate(times):
    sam.set_time(t)
    p0, p1 = sam.prob_evol()
    axs[i].plot(np.arange(0, len(p0))*(sam.get_time()/len(p0)), p0, label=rf'Joint prob $P_0(n,t)$', color='blue')
    axs[i].plot(np.arange(0, len(p1))*(sam.get_time()/len(p1)), p1, label=rf'Joint prob $P_1(n,t)$', color='red')
    # Calculate conditional probabilities without calling the function for efficiency
    p0_cond = p0/(p0+p1)
    p1_cond = p1/(p0+p1)
    axs[i].plot(np.arange(0, len(p0))*(sam.get_time()/len(p0)), p0_cond, label=rf'Cond prob $P_0(t|n)$', color='blue', linestyle='--')
    axs[i].plot(np.arange(0, len(p1))*(sam.get_time()/len(p1)), p1_cond, label=rf'Cond prob $P_1(t|n)$', color='red', linestyle='--')

plt.tight_layout()
plt.savefig(f'../images/conditional_probabilities_Tm_{sam.get_temp_meter()}.png')
print("Conditional probabilities plotted")

fig, ax = plt.subplots()
time = 0.9
sam.set_time(time)

meas_levels = np.arange(0, sam.get_total_levels()+1)
joint_probs = np.zeros_like(meas_levels, dtype=np.float64)
joint_probs_0 = np.zeros_like(meas_levels, dtype=np.float64)
cond_probs = np.zeros_like(meas_levels, dtype=np.float64)
for i in range(len(meas_levels)):
    sam.set_n(i) # Set the meter level to i
    p0_n, p1_n = sam.joint_probability(sam.get_n(), sam.get_time())
    joint_probs[i] = p1_n
    joint_probs_0[i] = p0_n
    cond_probs[i] = p1_n/(p0_n+p1_n)
ax.scatter(meas_levels, joint_probs, s=10, color='red', label='P_1(n,t)')
ax.scatter(meas_levels, cond_probs, s=10, color='blue', label='P(1|n,t)')
ax.scatter(meas_levels, joint_probs_0, s=10, color='green', label='P_0(n,t)')
ax.set_xlabel('Meter Level')
ax.set_ylabel('Probability')
ax.legend()
ax.set_title(f'Prob of i given meter in n time t={time:.3f}')
plt.savefig(f'../images/joint_probabilities_Tm_{sam.get_temp_meter()}_tm_{sam.get_time()}.png')
print("Joint probabilities plotted")

# Plot conditional entropy
times = np.array([0.5, 0.75, 1.0])
fig, axs = plt.subplots(len(times),1)
for i,t in enumerate(times):
    cond_entropy = np.zeros(sam.get_total_levels(), dtype=np.float64)
    p0_cond = np.zeros(sam.get_total_levels(), dtype=np.float64)
    p1_cond = np.zeros(sam.get_total_levels(), dtype=np.float64)
    for n in range(sam.get_total_levels()):
        #cond_entropy[n] = sam.conditional_entropy(n=n, time=t)
        p0, p1 = sam.conditional_probability(n, t)
        p0_cond[n] = p0
        p1_cond[n] = p1
        cond_entropy[n] = -(p1*np.log(p1) + p0*np.log(p0))
    axs[i].scatter(np.arange(0, sam.get_total_levels()), cond_entropy, label=f'$S_n(t={t})$')
    axs[i].set_xlabel('Meter Level')
    axs[i].set_ylabel('Conditional Entropy')
    axs[i].set_title(f'Conditional Entropy as a function of meter level at time t={t}')
    axs[i].legend()
plt.tight_layout()
plt.savefig(f'../images/conditional_entropy_Tm_{sam.get_temp_meter()}.png')
print("Conditional entropy plotted")

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
ax.plot(times, cond_entropy, color='red', label=f'Conditional Entropy, n={sam.get_n()}')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Entropy')
ax.set_title('Entropy as a function of time')
plt.savefig(f'../images/entropy_Tm_{sam.get_temp_meter()}.png')
print("Entropy plotted")

# Plot mutual information as a function of time
mut_info = np.zeros_like(times)
fig, ax = plt.subplots()
#for i, t in enumerate(times):
#    #sam.set_time(t)
#    mut_info[i] = sam.mutual_information(time = t)
mut_info = -(entropy - entropy[0])
ax.plot(times, mut_info, color='blue', label='Mutual Information')
ax.set_xlabel('Time')
ax.set_ylabel('Mutual Information')
ax.set_title('Mutual information between system and meter as a function of time')
plt.savefig(f'../images/mutual_information_Tm_{sam.get_temp_meter()}.png')
print("Mutual information plotted")

# Plot observer information at time t=0.9 as a function of meter level
fig, ax = plt.subplots()
sam.set_time(0.9)
for n in range(sam.get_total_levels()):
    sam.set_n(n)
    obs_info = sam.observer_information()
    ax.scatter(n, obs_info, color='blue', label='Observer Information')
ax.set_xlabel('Meter Level')
ax.set_ylabel('Observer Information')
ax.set_title('Observer Information as a function of meter level')
plt.savefig(f'../images/observer_information_Tm_{sam.get_temp_meter()}.png')
print("Observer information plotted")