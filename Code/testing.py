from SystemAndMeter import SystemAndMeter
import numpy as np
import matplotlib.pyplot as plt

temp_meter = 50
temp_system = 1
omega_meter = 2*np.pi
coupling = 5
init_system_state = (1/2, 1/2)
sam = SystemAndMeter(temp_meter=temp_meter, temp_system=temp_system, omega_meter=omega_meter,\
                     coupling=coupling, init_system_state=init_system_state)
print(sam.get_total_levels())
sam.set_time(10/omega_meter)
print(sam.get_time())

# Checking Sum_m |<n|U(t)|m>|^2 for a few random times and all n
print("Checking Sum_m |<n|U(t)|m>|^2 for a few random times and all n ...")
rng = np.random.default_rng(seed=42)
times = [0, 0.5, 0.75, 1.0]
#times = rng.uniform(0,1, 4)
times = np.sort(times)
fig, axs = plt.subplots(len(times), 1, figsize=(10, 8))
QHO_levels = np.arange(0, sam.get_total_levels())
shift_factors = np.zeros_like(QHO_levels, dtype=float)
for i, time in enumerate(times):
    for n in QHO_levels:
        p = 0
        for m in QHO_levels:
            p += sam.shift_factor(n, m, time)
        shift_factors[n] = p
    axs[i].scatter(QHO_levels, shift_factors)
    axs[i].set_ylim(-0.1, 1.1)
    axs[i].set_xlabel('n')
    axs[i].set_ylabel('Shift Factor')
    axs[i].set_title(f't = {time}')
plt.tight_layout()
print("... completed")

# Checking Sum_n Sum_m P_m |<n|U(t)|m>|^2 for the same random times
print("Checking Sum_n Sum_m P_m |<n|U(t)|m>|^2 for the same random times ...")
P_m = np.arange(0, sam.get_total_levels(), dtype=float)
P_m = sam.population_distribution(P_m)
for i, time in enumerate(times):
    res = np.sum(P_m * shift_factors)
    print(f"Sum_n Sum_m P_m |<n|U(t)|m>|^2 at t = {time:.3f} is {res}")
print("... completed")


#Checking Sum_t (<n|U(t)|m>) for a few random n and all times
print("Checking Sum_t (<n|U(t)|m>) for a few random n and all times ...")
msmt_levels = [0,1,2, 3, 4]
#msmt_levels = np.random.randint(0, sam.get_total_levels(), 5)
msmt_levels = np.sort(msmt_levels)
fig, axs = plt.subplots(len(msmt_levels),1, figsize=(10, 8))
plt.title('Shift factors for given n as a function of time')
time = np.linspace(0, 10/omega_meter, 100)
for i, n in enumerate(msmt_levels):
    shift_factors = np.zeros_like(time)
    shift_factors2 = np.zeros_like(time)
    for t in time:
        p = 0
        p2 = 0
        for m in QHO_levels:
            p += sam.shift_factor(n, m, t)
            p2 += sam.population_distribution(m) * sam.shift_factor(n, m, t)
        shift_factors[i] = p
        shift_factors2[i] = p2
    #axs[i].plot(time, shift_factors, label=rf"$\sum_m |\langle n|U(t)|m\rangle |^2$")
    axs[i].plot(time, shift_factors2, label=rf"$\sum_m P_m|\langle n|U(t)|m\rangle |^2$", linestyle='--')
    axs[i].legend()
   # axs[i].set_ylim(-0.1, 1.1)
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Shift Factor')
    axs[i].set_title(f'n = {n}')
    print(f"Shift factor as function of time for n = {n} completed")
plt.tight_layout()
print("... completed")

# Plot the individual terms of the shift factors for a few given n and t
print("Checking the individual matrix elements of D ...")
shift_factors = np.zeros_like(QHO_levels, dtype=float)
levels = 3
msmt_levels = np.sort(rng.integers(0, sam.get_total_levels(), levels))
msmt_levels = [0, 1, 2, 3]
times = rng.uniform(0, 1, len(msmt_levels))
time = rng.uniform(0, 1, 1)[0]
fig, axs = plt.subplots(levels, 1, figsize=(10, 8))
for i in range(levels):
    n = msmt_levels[i]
    t = time#s[i]
    for m in QHO_levels:
        shift_factors[m] = sam.shift_factor(n, m, t)
    p = sam.population_distribution(QHO_levels)
    axs[i].plot(QHO_levels, p*shift_factors)
    axs[i].scatter(QHO_levels, p*shift_factors)
    #axs[i].scatter(QHO_levels, shift_factors, color='red')
    #axs[i].set_ylim(-0.1, 1.1)
    axs[i].set_xlabel('m')
    axs[i].set_ylabel(rf'$\langle {n}|U({t:.2f})|m\rangle$')
    axs[i].set_title(f'n = {n}, t = {t:.2f}')
plt.tight_layout()
print("... completed")

# Checking that the population distribution is normalized
print("Checking that the population distribution is normalized ...")
p = sam.population_distribution(QHO_levels)
print(f"Sum of population distribution: {np.sum(p)}")
print("... completed")

# Checking the conditional probabilities for all n at a few random times
print("Checking the conditional probabilities for all n at a few random times ...")
times = rng.uniform(0, 1, 4)
times = np.sort(times)
fig, axs = plt.subplots(len(times), 1, figsize=(10, 8))
for i, t in enumerate(times):
    p0_cond = np.zeros_like(QHO_levels, dtype=float)
    p1_cond = np.zeros_like(QHO_levels, dtype=float)
    for n in QHO_levels:
        p0, p1 = sam.conditional_probability(n, t)
        p0_cond[n] = p0
        p1_cond[n] = p1
    axs[i].scatter(QHO_levels, p0_cond, color='green', label=rf'p(0|n,t={t:.2f})')
    axs[i].scatter(QHO_levels, p1_cond, color='red', label=rf'p(1|n,t={t:.2f})')
    axs[i].plot(QHO_levels, p0_cond+p1_cond, color='blue', label=rf'p(0|n,t={t:.2f})+p(1|n,t={t:.2f})')
    axs[i].set_xlabel(rf'Meter level $n$')
    axs[i].set_ylabel('Probability')
    axs[i].set_title(f'Conditional probabilities at time t = {t:.2f}')
    axs[i].legend()
plt.tight_layout()
print("... completed")

# Checking the conditional entropy for all n at a few random times
print("Checking the conditional entropy for all n at a few random times ...")
fig, axs = plt.subplots(len(times), 1, figsize=(10, 8))
for i, t in enumerate(times):
    p0_cond = np.zeros_like(QHO_levels, dtype=float)
    p1_cond = np.zeros_like(QHO_levels, dtype=float)
    for n in QHO_levels:
        #cond_entropy[n] = sam.conditional_entropy(n=n, time=t)
        p0, p1 = sam.conditional_probability(n, t)
        p0_cond[n] = p0
        p1_cond[n] = p1
    axs[i].scatter(QHO_levels, -(p1_cond*np.log(p1_cond) + p0_cond*np.log(p0_cond)), color='blue', label=rf'$S_n(t={t:.2f})$')
    axs[i].scatter(QHO_levels, -p1_cond*np.log(p1_cond), color='red', label=rf'$S_1(n,t={t:.2f})$')
    axs[i].scatter(QHO_levels, -p0_cond*np.log(p0_cond), color='green', label=rf'$S_0(n,t={t:.2f})$')
    axs[i].set_xlabel(rf'Meter level $n$')
    axs[i].set_ylabel('Conditional Entropy')
    axs[i].set_title(f'Conditional entropy at time t = {t:.2f}')
    axs[i].legend()
print("... completed")
plt.tight_layout()
plt.show()

