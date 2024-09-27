
import numpy as np
#import scipy as sp # Not currently used but can be uncommented if needed
from scipy.special import factorial, assoc_laguerre

hbar = 1.0 # sp.constants.hbar
kB = 1.0 # sp.constants.k

class SystemAndMeter:
    def __init__(self, temp_meter, temp_system, omega_meter, coupling=1, time=True,
                 total_levels=False, init_system_state =(1/2, 1/2), measurement_state = 0):
        """Initializes the system and meter with the given parameters.

        Args:
            temp_meter (float): The temperature of the meter.
            temp_system (float): The temperature of the system.
            omega_meter (float): The angular frequency of the meter.
            coupling (float, optional): The coupling strength between meter and system. Defaults to 1.
            time (bool or float, optional): The interaction time t_m. If true uses normalized time t=1*omega_meter. Defaults to True.
            total_levels (bool or int, optional): The total number of energy levels in the meter. If False automatically chooses 10*ceil(k_B T/ hbar omega). Defaults to False.
            init_system_state (tuple, optional): The initial state of the system. Defaults to (1/2, 1/2).
            measurement_state (int, optional): The state of the meter to use for measurements. Defaults to 0.
        """        
        
        if time == True:
            time = 1/omega_meter
        self.T_m = temp_meter
        self.temp_system = temp_system
        self.omega_meter = omega_meter
        self.g = coupling
        self.time = time
        self.beta = 1/(self.T_m*kB)
        self.tls_state = init_system_state
        self.n = measurement_state
        self.gamma = hbar*self.omega_meter
        if total_levels == False:
            self.total_levels = int((self.beta/self.gamma))+1
        else:
            self.total_levels = int(total_levels)
        self.meter_state = self.calc_meter_state()

    def calc_meter_state(self):
        """Generates the Gibbs-Boltzmann distribution of the meter.

        Returns:
            ndarray: The population in each energy level of the meter. Index 0 corresponds to the ground state.
        """
        # Returns the Gibbs-Boltzmann distribution of the meter
        levels = np.arange(0, self.total_levels+1)
        pop = self.population_distribution(levels)
        return pop
    
    def population_distribution(self, n):
        """Returns the Gibbs-Boltzmann probability at a given energy level n and the temperature of the meter.
            Accepts vectorized inputs.

        Args:
            n (int): The energy level of the meter to calculate the probability for.

        Returns:
            float or ndarray: The probability of the meter being in energy level n.
        """
        beta = self.beta
        gamma = self.gamma
        N = self.total_levels
        #p_n = (1-np.exp(-gamma*beta))*np.exp(-gamma*beta*n)/(np.exp(-gamma*beta*(m+1))-1)
        #p_n = (1-np.exp(-gamma*beta))*np.exp(-gamma*beta*n)/(1-np.exp(-gamma*beta*(N+1)))
        p_n = (1-np.exp(-beta*(gamma+0.5)))*np.exp(-beta*(gamma+0.5)*n)/(1-np.exp(-beta*(gamma+0.5)*(N+1)))
        return p_n

    def joint_probability(self, n, t):
        """Returns the joint probabilities of the system and meter being in a certain state at time t.

        Args:
            n (int): The energy level of the meter to condition on.
            t (float): The time to calculate the joint probabilities at.

        Returns:
            float: The joint probabilities of 0,n and 1,n.
        """
        g = self.g
        omega = self.omega_meter
        gamma = self.gamma
        #alpha = -g/np.sqrt(2*omega)*(np.sin(omega*t) +1j*(np.cos(omega*t) -1))
        alpha = -g/np.sqrt(gamma)*(np.sin(omega*t) -1j*(1-np.cos(omega*t)))
        # The joint probability of the system and meter being in a certain state
        # at time t
        p0_n = self.tls_state[0]*self.population_distribution(n)
        p1_n = 0
        for m in range(0, self.total_levels+1):
            if m >= n:
                p1_n += self.population_distribution(m)*np.abs(np.exp(-np.abs(alpha)**2/2)*\
                                                               np.sqrt(factorial(n)/factorial(m))*\
                                                               (alpha)**(m-n)*\
                                                                assoc_laguerre(np.abs(alpha)**2, n, np.abs(m-n)))**2
            else:
                p1_n += self.population_distribution(m)*np.abs(np.exp(-np.abs(alpha)**2/2)*\
                                                                np.sqrt(factorial(m)/factorial(n))*\
                                                                (-np.conjugate(alpha))**(n-m)*\
                                                                 assoc_laguerre(np.abs(alpha)**2, m, np.abs(m-n)))**2
        p1_n *= self.tls_state[1]
        return p0_n, p1_n
        
    def conditional_probability(self, n, t):
        """Calculates the conditional probabilities of the system being in state 0 or 1 given the meter is in state n at time t.

        Args:
            n (int): The energy level of the meter to condition on.
            t (float): The time to calculate the conditional probabilities at.

        Returns:
            float: First value is the conditional probability of the system being in state 0 given the meter is in state n at time t.
                    Second value is the conditional probability of the system being in state 1 given the meter is in state n at time t.
        """
        # First get the joint probabilities
        p0_n, p1_n = self.joint_probability(n, t)
        # The conditional probability of the system being in state 0 given the meter is in state n
        # at time t
        p0_n_given = p0_n/(p0_n + p1_n)
        #p0_n_given = p0_n/self.population_distribution(n)
        # The conditional probability of the system being in state 1 given the meter is in state n
        # at time t
        p1_n_given = p1_n/(p0_n + p1_n)
        #p1_n_given = p1_n/self.population_distribution(n)
        return p0_n_given, p1_n_given

    def prob_evol(self):
        """Time evolution of the conditional probabilities of the system being in state 0 or 1

        Returns:
            ndarray: First array is the conditional probability of the system being in state 0 given the meter is in state n at time t.
                    Second array is the conditional probability of the system being in state 1 given the meter is in state n at time t.
        """
        # Eigenstate of the meter to measure
        n = self.n
        # Initialize the system and meter states
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

    def joint_prob_evol(self):
        """Time evolution of the joint probabilities of the system being in state 0 or 1 and the meter being in state n.

        Returns:
            ndarray: First array is the joint probability of the system being in state 0 and the meter being in state n.
                    Second array is the joint probability of the system being in state 1 and the meter being in state n.
        """
        # Eigenstate of the meter to measure
        n = self.n
        # Initialize the system and meter states
        # Time step
        dt = self.time/1000
        # Time array
        time = np.arange(0, self.time, dt)
        # Initialize the probability arrays
        p0_n = np.zeros(len(time))
        p1_n = np.zeros(len(time))
        # Loop through time
        for i,t in enumerate(time):
            # Compute the joint probabilities at time t
            p0, p1 = self.joint_probability(n, t)
            p0_n[i] = p0
            p1_n[i] = p1
        return p0_n, p1_n

    def conditional_entropy(self, time=False, n=False):
        """Calculates the conditional entropy of the system given the meter at time t.

        Args:
            time (float, optional): The time to calculate the conditional entropy at. Defaults to self.time if not set.
            n (int, optional): The energy level of the meter to condition on. Defaults to self.n if not set.

        Returns:
            float: The conditional entropy of the system given the meter at time t.
        """
        if not time:
            time = self.time
        # Eigenstate of the meter to measure in
        n = self.n
        # Get the conditional probabilities P(0|n,t) and P(1|n,t)
        p0, p1 = self.conditional_probability(n, time)
        # Calculate the conditional entropy
        S_n = -kB*(p0*np.log(p0) + p1*np.log(p1))
        return S_n
    
    def entropy(self, time=False):
        """Calculates the entropy of the system at time t.

        Args:
            time (float, optional): The time to calculate the entropy at. Defaults to self.time if not set.

        Returns:
            float: The entropy of the system at time t.
        """
        if not time:
            time = self.time
        N = self.total_levels
        S = 0
        for n in range(0, N+1):
            S_n = self.conditional_entropy(time, n)
            p0_n, p1_n = self.joint_probability(n, time)
            S += (p0_n + p1_n)*S_n
        return S
    
    def mutual_information(self, time=False):
        """Calculates the mutual information between the system and meter at time t.

        Args:
            time (float, optional): The time to calculate the mutual information at. Defaults to self.time if not set.

        Returns:
            float: The mutual information between the system and meter at time t.
        """
        if not time:
            time = self.time
        S_0 = self.entropy(time=0)
        S_t = self.entropy(time)
        return S_0 - S_t
    
    def observer_information(self, time=False, n=False):
        """Calculates the observers information about the system after measurement in n at time t.

        Args:
            time (float, optional): The time to calculate the observer information at. Defaults to self.time if not set.
            n (int, optional): The energy level of the meter to condition on. Defaults to self.n if not set.

        Returns:
            float: The observer information between the system and meter at time t.
        """
        if not time:
            time = self.time
        if not n:
            n = self.n
        p0_n, p1_n = self.joint_probability(n, time)
        p_n = p0_n + p1_n
        I_O = -kB*(p_n*np.log(p_n) + (1-p_n)*np.log(1-p_n))
        return I_O

    # Functions to set the parameters of the system and meter
    def set_temp_system(self, temp_system):
        """Setter function for the temperature of the system.

        Args:
            temp_system (float): Temperature of the system.
        """
        self.temp_system = temp_system
    def set_temp_meter(self, temp_meter):
        """Setter function for the temperature of the meter.

        Args:
            temp_meter (float): Temperature of the meter.
        """
        self.T_m = temp_meter
    def set_coupling(self, coupling):
        """Setter function for the coupling strength between the system and meter.

        Args:
            coupling (float): Coupling strength between the system and meter.
        """
        self.g = coupling
    def set_time(self, time):
        """Setter function for the interaction time between the system and meter.

        Args:
            time (float): Time of interaction between the system and meter.
        """
        self.time = time
    def set_omega(self, omega_meter):
        """Setter function for the angular frequency of the meter.

        Args:
            omega_meter (float): Angular frequency of the meter.
        """
        self.omega_meter = omega_meter
    def set_n(self, n):
        """Setter function for the state of the meter to measure.

        Args:
            n (int): State of the meter to measure.
        """
        self.n = n
    def set_tls_state(self, tls_state):
        """Setter function for the initial state of the system.

        Args:
            tls_state (tuple): Initial state of the system.
        """
        self.tls_state = tls_state
    def set_total_levels(self, total_levels):
        """Setter function for the total number of energy levels in the meter.

        Args:
            total_levels (int): Total number of energy levels in the meter.
        """
        self.total_levels = total_levels
    def update_params(self):
        """
            Updates the hidden parameters of the system and meter. I.e. the beta and gamma values.
        """
        self.gamma = hbar*self.omega_meter
        self.beta = 1/(self.T_m*kB)
    
    #--------- Functions to update the hidden parameters of the system and meter ------------
    def update_total_levels(self):
        """
            Updates the total number of energy levels in the meter.
        """
        self.total_levels = 10*int((1/(self.beta*self.gamma)))+1
        self.meter_state = self.calc_meter_state()
    def full_update(self):
        """
            Updates all the hidden parameters of the system and meter.
            Might be advisable to run this whenever a parameter is changed but it's up to the user.
        """
        self.update_params()
        self.update_total_levels()
    #-----------------------------------------------------------------------------------------

    #-------- Functions to get the parameters of the system and meter ------------------------
    def get_temp_system(self):
        """Getter function for the temperature of the system.

        Returns:
            float: Temperature of the system.
        """
        return self.temp_system
    def get_temp_meter(self):
        """Getter function for the temperature of the meter.

        Returns:
            float: Temperature of the meter.
        """
        return self.T_m
    def get_coupling(self):
        """Getter function for the coupling strength between the system and meter.

        Returns:
            float: Coupling strength between the system and meter.
        """
        return self.g
    def get_time(self):
        """Getter function for the interaction time between the system and meter.

        Returns:
            float: Time of interaction between the system and meter.
        """
        return self.time
    def get_omega(self):
        """Getter function for the angular frequency of the meter.

        Returns:
            float: Angular frequency of the meter.
        """
        return self.omega_meter
    def get_n(self):
        """Getter function for the state of the meter to measure.

        Returns:
            int: State of the meter to measure.
        """
        return self.n
    def get_tls_state(self):
        """Getter function for the initial state of the system.

        Returns:
            tuple: Initial state of the system.
        """
        return self.tls_state
    def get_total_levels(self):
        """Getter function for the total number of energy levels in the meter.

        Returns:
            int: Total number of energy levels in the meter.
        """
        return self.total_levels
    def get_beta(self):
        """Getter function for the beta value.

        Returns:
            float: Beta value.
        """
        return self.beta
    def get_gamma(self):
        """Getter function for the gamma value.

        Returns:
            float: Gamma value.
        """
        return self.gamma
   
