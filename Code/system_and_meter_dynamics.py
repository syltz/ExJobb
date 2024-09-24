import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import factorial, assoc_laguerre
import cmath

# Throughout this code, the meter is assumed to be a harmonic oscillator
# and the system is assumed to be a two-level system.
# I also use mass = 1, hbar = 1, and kB = 1 for simplicity.

class SystemAndMeter:
    # Initilizes with temperature, angular frequency, coupling strength, and time
    # If time is True then use a normalized time 1/omega
    # Otherwise use the time given
    # Parameters:
    # temp_meter: float, temperature of the meter (K)
    # temp_system: float, temperature of the system (K)
    # omega_meter: float, angular frequency (rad/time)
    # coupling: float, coupling strength (dimensionless)
    # time: float, time (arbitrary units) or True, default is True meaning t=1/omega
    # total_levels: int, total number of energy levels in the meter before truncating
    # init_system_state: tuple, initial state of the TLS
    # measurement_state: int, state of the meter to measure
    def __init__(self, temp_meter, temp_system, omega_meter, coupling=1, time=True,
                 total_levels=10, init_system_state =(1/2, 1/2), measurement_state = 0):
        if time == True:
            time = 1/omega_meter
        self.T_m = temp_meter
        self.temp_system = temp_system
        self.omega_meter = omega_meter
        self.g = coupling
        self.time = time
        self.total_levels = total_levels
        self.beta = 1/self.T_m
        self.tls_state = init_system_state
        self.meter_state = self.meter_state()
        self.n = measurement_state

    def meter_state(self):
        # Returns the Gibbs-Boltzmann distribution of the meter
        levels = np.arange(0, self.total_levels+1)
        pop = self.population_distribution(levels)
        return pop
    def population_distribution(self, n):
        # Returns the Gibbs-Boltzmann distribution at the given temperature and
        # energy levels (n) for the meter.
        # Energies of the meter
        # Canonical partition function
        # Energies are hbar*omega*(n+1/2) but hbar = 1
        omega = self.omega_meter
        beta = self.beta
        m = self.total_levels
        p_n = (1-np.exp(omega*beta))*np.exp(omega*beta*n)/(np.exp(-omega*beta*(m+1))-1)
        return p_n
    def conditional_probability(self, n, t):
        # Conditional probabilies of the system being in state 0 or 1 given the meter is in state n
        # at time t.
        # alpha parameter
        g = self.g
        omega = self.omega_meter
        alpha = -g/np.sqrt(2*omega)*(np.sin(omega*t) +1j*(np.cos(omega*t) -1))
        # The joint probability of the system and meter being in a certain state
        # at time t
        p0_n = self.tls_state[0]*self.population_distribution(n)
        p1_n = 0
        for m in range(0, self.total_levels+1):
            if t==0 and np.abs(alpha) == 0:
                alpha_term = -g/np.sqrt(2*omega)
            else:
                alpha_term = alpha
            #if m <= n:
            #    p1_n += self.population_distribution(m)*np.abs(np.sqrt(factorial(m)/factorial(n))*\
            #    alpha_term**(n-m)*np.exp(-np.abs(alpha)**2/2)*assoc_laguerre(np.abs(alpha_term)**2, m, n-m))**2
            #else:
            #    #p1_n += self.population_distribution(m)*np.abs(np.sqrt(factorial(m)/factorial(n))*\
            #    #                                               alpha_term**(n-m)*np.exp(-np.abs(alpha)**2/2)*\
            #    #                                                (-np.abs(alpha_term)**2)**(m-n)*factorial(n)/factorial(m)*assoc_laguerre(np.abs(alpha_term)**2, n-m, m-n))**2
            #    p1_n += self.population_distribution(m)*np.abs(np.sqrt(factorial(n)/factorial(m))*\
            #                                                   alpha_term**(n-m)*(-1*np.abs(alpha_term)**2)**(m-n)*\
            #                                                    np.exp(-np.abs(alpha)**2/2)*assoc_laguerre(np.abs(alpha_term)**2, n, m-n))**2
            if m <= n:
                p1_n += self.population_distribution(m)*np.abs(np.sqrt(factorial(m)/factorial(n))*\
                                                               np.exp(-np.abs(alpha)**2/2)*(alpha)**(n-m)*assoc_laguerre(np.abs(alpha)**2, m, n-m))**2
            else:
                p1_n += self.population_distribution(m)*np.abs(np.sqrt(factorial(n)/factorial(m))*\
                                                               np.exp(-np.abs(alpha)**2/2)*(-np.conjugate(alpha))**(m-n)*\
                                                                assoc_laguerre(np.abs(alpha)**2, n, m-n))**2
        p1_n *= self.tls_state[1]
        # The conditional probability of the system being in state 0 given the meter is in state n
        # at time t
        p0_n_given = p0_n/(p0_n + p1_n)
        # The conditional probability of the system being in state 1 given the meter is in state n
        # at time t
        p1_n_given = p1_n/(p0_n + p1_n)
        return p0_n_given, p1_n_given

    def prob_evol(self):
        # Time evolution of the conditional probabilities of the system being in state 0 or 1
        # given the meter is in state n
        # Eigenstate of the meter to measure
        n = self.n
        # Initialize the system and meter states
        meter_state = self.meter_state
        # Time step
        dt = self.time/1000
        # Time array
        time = np.arange(0, self.time, dt)
        # Initialize the probability arrays
        p0_n_given = np.zeros(len(time))
        p1_n_given = np.zeros(len(time))
        # Loop through time
        for i,t in enumerate(time):
            # Compute the conditional probabilities at time t
            p0, p1 = self.conditional_probability(n, t)
            p0_n_given[i] = p0
            p1_n_given[i] = p1
        return p0_n_given, p1_n_given

    def dynamics(self):
        # Time evolution of the system and meter
        # Initialize the system and meter states
        system_state = self.tls_state
        meter_state = self.meter_state
        # Time step
        dt = self.time/1000
        # Time array
        time = np.arange(0, self.time, dt)
        # Loop through time
        for t in time:
            # Update the meter state
            for n in range(0, self.total_levels+1):
                p0_n_given, p1_n_given = self.conditional_probability(n, t)
                meter_state[n] = p0_n_given*system_state[0] + p1_n_given*system_state[1]
            # Update the system state
            system_state = np.array([np.sum(meter_state), np.sum(meter_state)])
        return system_state, meter_state

    # Functions to set the parameters of the system and meter
    def set_temp_system(self, temp_system):
        self.temp_system = temp_system
    def set_temp_meter(self, temp_meter):
        self.T_m = temp_meter
        self.beta = 1/self.T_m
    def set_coupling(self, coupling):
        self.g = coupling
    def set_time(self, time):
        self.time = time
    def set_omega(self, omega_meter):
        self.omega_meter = omega_meter

sam = SystemAndMeter(temp_meter=1, temp_system=1, omega_meter=2*np.pi, time=10/(2*np.pi), init_system_state=(1/2, 1/2), measurement_state=1, coupling=1)
p0, p1 = sam.prob_evol()
plt.scatter(np.arange(0, len(p0)), p0, s=1, label='p0')
plt.scatter(np.arange(0, len(p1)), p1, s=1, label='p1')
plt.plot(p0+p1, label='p0+p1', color='black')
plt.xlabel('Time index')
plt.ylabel('Probability')
plt.legend()
plt.show()
