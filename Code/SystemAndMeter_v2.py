import numpy as np
import scipy as sp # Not currently used but can be uncommented if needed
from scipy.special import factorial, assoc_laguerre

kB = 1e3*sp.constants.physical_constants['Boltzmann constant in eV/K'][0] # Boltzmann constant in meV/K
hbar = 1e3*sp.constants.physical_constants['reduced Planck constant in eV s'][0]# Reduced Planck constant in meV s

class SystemAndMeter:
    def __init__(self, T_S=300, x=1, Q_S=1, Q_M=1, P=1, tau = 0.5, msmt_state=0):
        """Initializes the system and meter with the given parameters.
        Ideally, we use the dimensionless parameters Q_S, Q_M, and P to set the energy scales of the system and meter.
        However, you can set the parameters directly if you want via the seter functions, but this requires
        a bit more care.

        Args:
            T_S (float): The temperature of the system, T_S. Default is 300 K.
            x (float): The normalized temperature of the meter, T_M = x*T_S. Default is 1.
            Q_S (float): Dimensionless scaling parameter for the TLS energy, delta_E = Q_S*kB*T_S. Default is 1.
            Q_M (float): Dimensionless scaling parameter for the meter energy, hbar*omega = Q_M*delta_E. Default is 1.
            P (float): Dimensionless parameter that sets coupling strength and mass, g^2*m/2 = P*delta_E. Default is 1.
            tau (float): The normalized time of interaction between the system and meter, t = tau*2*pi/omega. Default is 0.5.
            msm_state (int): The energy level of the meter to measure. Default is 0.

        """        
        self.T_S = T_S
        self.x = x
        self.Q_S = Q_S
        self.Q_M = Q_M
        self.P = P
        self.n = msmt_state
        self.tau = tau
        self.update_params()
        self.update_total_levels()
        
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
            Accepts vectorized inputs. Assumes total_levels chosen so that N >> k_B T/hbar omega.

        Args:
            n (int): The energy level of the meter to calculate the probability for.

        Returns:
            float or ndarray: The probability of the meter being in energy level n.
        """
        # If T_m = 0 then beta = inf which is not a number, so we need to check for this.
        if self.T_m == 0 and n > 0:
            # lim p(n) = 0 as beta -> inf for n > 0 and N > 0.
            return 0
        elif self.T_m == 0 and n == 0:
            # lim p(0) = 1 as beta -> inf for n = 0 and N > 0.
            return 1
        beta = self.beta
        omega = self.omega_meter
        Z_prime = 1/(1-np.exp(-hbar*omega/(kB*self.T_m)))
        p_n = np.exp(-beta*hbar*omega*n)/Z_prime
        return p_n
    
    def shift_factor(self, n, m, t):
        """Calculates <n|D(t)|m> where D(t) is the displacement operator using Cahill and Glauber's formula.
        Should not be called directly. Used in the calculation of the joint probabilities.
        You can call it directly, but just know what you're doing.

        Args:
            n (int): The measurement state of the meter.
            m (int): An energy level of the meter.
            t (float): The system time to calculate the shift factor at.

        Returns:
            float: <n|D(t)|m>
        """

        g = self.g
        omega = self.omega_meter
        mass = self.mass
        alpha = g*np.sqrt(mass/(2*hbar*omega))*(np.sin(omega*t) -1j*(np.cos(omega*t)-1))
        if n >= m:
            #d =  np.abs(np.exp(-np.abs(alpha)**2/2)*np.sqrt(factorial(m)/factorial(n))*\
            #              alpha**(n-m)*assoc_laguerre(np.abs(alpha)**2, m, np.abs(m-n)))**2
            d = np.abs(np.exp(-np.abs(alpha)**2/2)*np.sqrt(self.factorial_ratio(m,n))*\
                            alpha**(n-m)*assoc_laguerre(np.abs(alpha)**2, m, np.abs(m-n)))**2
        else:
            #d =  np.abs(np.exp(-np.abs(alpha)**2/2)*np.sqrt(factorial(n)/factorial(m))*\
            #              (-np.conjugate(alpha))**(m-n)*assoc_laguerre(np.abs(alpha)**2, n, np.abs(m-n)))**2
            d = np.abs(np.exp(-np.abs(alpha)**2/2)*np.sqrt(self.factorial_ratio(n,m))*\
                            (-np.conjugate(alpha))**(m-n)*assoc_laguerre(np.abs(alpha)**2, n, np.abs(m-n)))**2
        return d
    
    def factorial_ratio(self, num, den):
        """Calculates the ratio of two factorials assuming the denominator is greater than the numerator.

        Args:
            num (int): Numerator.
            den (int): Denominator.

        Returns:
            float: n!/(m!)
        """
        ratio = 0
        if num == den:
            ratio = 1
        else: 
            # Calculate the ratio of factorials using the log
            new_den = np.sum(np.log(np.arange(num+1, den+1), dtype=np.float64))
            # Check if new_den is infinity or if it's greater than the log of the maximum float value
            if new_den == np.inf or new_den > np.log(np.finfo(np.float64).max):
                ratio = 0
            else:
                ratio = 1/np.exp(new_den)
        return ratio
    
    def joint_probability(self, n=None, t=None):
        """Returns the joint probabilities of the system and meter being in a certain state at time t.

        Args:
            n (int, optional): The energy level of the meter to condition on. Defaults to self.n if not set.
            t (float, optional): The time to calculate the joint probabilities at. Defaults to self.time if not set.

        Returns:
            float: The joint probabilities of 0,n and 1,n.
        """
        # The joint probability of the system and meter being in a certain state
        # at time t
        if (n==None):
            n = self.n
        if (t==None):
            t = self.time
        p0_n = self.tls_state[0]*self.population_distribution(n)
        p1_n = 0
        for m in range(0, self.total_levels+1):
            p1_n += self.population_distribution(m)*self.shift_factor(n=n, m=m, t=t)

        p1_n *= self.tls_state[1]
        return p0_n, p1_n
        
    def conditional_probability(self, n=None, t=None):
        """Calculates the conditional probabilities of the system being in state 0 or 1 given the meter is in state n at time t.

        Args:
            n (int, optional): The energy level of the meter to condition on. Defaults to self.n if not set.
            t (float, optional): The time to calculate the conditional probabilities at. Defaults to self.time if not set.

        Returns:
            float: First value is the conditional probability of the system being in state 0 given the meter is in state n at time t.
                    Second value is the conditional probability of the system being in state 1 given the meter is in state n at time t.
        """
        if (n==None):
            n = self.n
        if (t==None):
            t = self.time
        # First get the joint probabilities
        p0_n, p1_n = self.joint_probability(n=n, t=t)
        # The conditional probability of the system being in state 0 given the meter is in state n
        # at time t
        if p0_n == 0:
            p0_n_given = 0
        else:
            p0_n_given = p0_n/(p0_n + p1_n)
        # The conditional probability of the system being in state 1 given the meter is in state n
        # at time t
        if p1_n == 0:
            p1_n_given = 0
        else:
            p1_n_given = p1_n/(p0_n + p1_n)
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
        dt = self.time/100
        # Time array
        time = np.arange(0, self.time, dt)
        # Initialize the probability arrays
        p0_n_given = np.zeros(len(time))
        p1_n_given = np.zeros(len(time))
        # Loop through time
        for i,t in enumerate(time):
            # Compute the conditional probabilities at time t
            p0, p1 = self.conditional_probability(n=n, t=t)
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
        dt = self.time/100
        # Time array
        time = np.arange(0, self.time, dt)
        # Initialize the probability arrays
        p0_n = np.zeros(len(time))
        p1_n = np.zeros(len(time))
        # Loop through time
        for i,t in enumerate(time):
            # Compute the joint probabilities at time t
            p0, p1 = self.joint_probability(n=n, t=t)
            p0_n[i] = p0
            p1_n[i] = p1
        return p0_n, p1_n

    def conditional_entropy(self, n=None, time=None):
        """Calculates the conditional entropy of the system given the meter at time t.

        Args:
            n (int, optional): The energy level of the meter to condition on. Defaults to self.n if not set.
            time (float, optional): The time to calculate the conditional entropy at. Defaults to self.time if not set.

        Returns:
            float: The conditional entropy of the system given the meter at time t.
        """
        if (time==None):
            time = self.time
        # Eigenstate of the meter to measure in
        if (n==None):
            n = self.n
        # Get the conditional probabilities P(0|n,t) and P(1|n,t)
        p0, p1 = self.conditional_probability(n=n, t=time)
        # Calculate the conditional entropy
        S_n = -(p0*np.log(p0) + p1*np.log(p1))
        return S_n
    
    def entropy(self, time=None):
        """Calculates the entropy of the system at time t.

        Args:
            time (float, optional): The time to calculate the entropy at. Defaults to self.time if not set.

        Returns:
            float: The entropy of the system at time t.
        """
        if (time==None):
            time = self.time
        N = self.total_levels
        S = 0
        for n in range(0, N+1):
            S_n = self.conditional_entropy(time=time, n=n)
            p0_n, p1_n = self.joint_probability(n, time)
            S += (p0_n + p1_n)*S_n
        return S
    
    def mutual_information(self, time=None):
        """Calculates the mutual information between the system and meter at time t.
            Basically just a wrapper for the entropy and conditional entropy functions.

        Args:
            time (float, optional): The time to calculate the mutual information at. Defaults to self.time if not set.

        Returns:
            float: The mutual information between the system and meter at time t.
        """
        if (time==None):
            time = self.time
        S_0 = self.entropy(time=0)
        S_t = self.entropy(time)
        return S_0 - S_t
    
    def observer_information(self, time=None, n=None):
        """Calculates the observers information about the system after measurement in n at time t.

        Args:
            time (float, optional): The time to calculate the observer information at. Defaults to self.time if not set.
            n (int, optional): The energy level of the meter to condition on. Defaults to self.n if not set.

        Returns:
            float: The observer information between the system and meter at time t.
        """
        if (time==None):
            time = self.time
        if (n==None):
            n = self.n
        p0_n, p1_n = self.joint_probability(n=n, t=time)
        p_n = p0_n + p1_n
        I_O = -kB*(p_n*np.log(p_n) + (1-p_n)*np.log(1-p_n))
        return I_O

    def work_extraction(self, time=None):
        """Calculates the work extracted from the system at time t.

        Args:
            time (float, optional): System time to calculate the work extracted at. Defaults to self.time if not set.

        Returns:
            float: The work extracted from the system at time t. sum_n P(n,t)*(P(1|n,t) - P(1|n,0))*delta_E
        """
        if (time==None):
            time = self.time
        delta_E = self.delta_E
        a, b = self.tls_state
        N = self.total_levels
        W = 0
        r = 10
        #for i in range(0, N):
        #    p0_n, p1_n = self.joint_probability(n=i, t=time) # P(0,n,t) and P(1,n,t)
        #    p_1_cond_t = p1_n/(p0_n + p1_n) # P(1|n,t)
        #    p1_cond_t0 = self.conditional_probability(n=i, t=0)[1] # P(1|n,0)
        #    W += (p0_n + p1_n)*delta_E*(p_1_cond_t - p1_cond_t0) # P(n,t)*(P(1|n,t) - P(1|n,0))
        for n in range(0, 2):
            p0_n, p1_n = self.joint_probability(n=n, t=time)
            W+=a*p1_n - b*p0_n
        W *= delta_E
        return W

    def work_measurement(self, time=None):
        """Returns the amount of work required to measure the meter at time t.

        Args:
            time (float, optional): The time to measure at. Defaults to self.time if not set.

        Returns:
            float: The work required to measure the meter at time t.
        """
        if time == None:
            time = self.time
        g = self.g
        omega = self.omega_meter
        m = self.mass   
        b = self.tls_state[1]
        return b*m*g**2*(1-np.cos(omega*time))

    def quality_factor(self, time=None):
        """Calculates the quality factor of the measurement at time t.

        Args:
            time (float, optional): The time to calculate the quality factor at. Defaults to self.time if not set.

        Returns:
            float: The quality factor of the measurement at time t.
        """
        if time == None:
            time = self.time
        W_ext = self.work_extraction(time)
        W_msmt = self.work_measurement(time)
        if W_ext==0:
            return 0
        else:
            return W_ext/W_msmt
    def quality_factor_info(self, time=None):

        if time == None:
            time = self.time
        mutual_info = self.mutual_information(time)
        W_ext = self.work_extraction(time)
        W_msmt = self.work_measurement(time)
        # Since we divide by W_ext and W_msmt, we need to check if they are zero
        # If they are zero, we set the quality factor to zero since this should also mean
        # that the mutual information is zero.
        if mutual_info == 0:
            return 0.0, 0.0
        else: 
            return mutual_info/W_ext, mutual_info/W_msmt



    # Functions to set the parameters of the system and meter
    def set_Q_S(self, Q_S):
        """Setter function for the dimensionless scaling parameter for the TLS energy.

        Args:
            Q_S (float): Dimensionless scaling parameter for the TLS energy.
        """
        self.Q_S = Q_S
        self.update_params()
    def set_Q_M(self, Q_M):
        """Setter function for the dimensionless scaling parameter for the meter energy.

        Args:
            Q_M (float): Dimensionless scaling parameter for the meter energy.
        """
        self.Q_M = Q_M
        self.full_update()
        #self.update_params()
    def set_P(self, P):
        """Setter function for the dimensionless parameter that sets coupling strength and mass.

        Args:
            P (float): Dimensionless parameter that sets coupling strength and mass.
        """
        self.P = P
        self.update_params()
    def set_x(self, x):
        """Setter function for the normalized temperature of the meter.

        Args:
            x (float): Normalized temperature of the meter.
        """
        self.x = x
        self.update_params()
        self.update_total_levels()
    def set_tau(self, tau):
        """Setter function for the normalized time of interaction between the system and meter.

        Args:
            tau (float): Normalized time of interaction between the system and meter.
        """
        self.tau = tau
        self.update_params()
    def set_T_S(self, T_S):
        """Setter function for the temperature of the system.

        Args:
            T_S (float): Temperature of the system.
        """
        self.T_S = T_S
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
    def set_time(self, tau):
        """Setter function for the interaction time between the system and meter.

        Args:
            time_norm (float): Normalized time of interaction between the system and meter.
                                t = tau*2*pi/omega
        """
        self.time = tau*2*np.pi/self.omega_meter
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
    def set_delta_E(self, delta_E):
        """Setter function for the energy difference between the two states of the system.

        Args:
            delta_E (float): Energy difference between the two states of the system.
        """
        self.delta_E = delta_E
    def update_params(self):
        """
            Updates the hidden parameters of the system and meter.
            The only parameters that we should set directly are the system temperature,
            the scaling parameters Q_S, Q_M, and P, the normalized temperature of the meter x,
            and the normalized time of interaction tau.
            The rest of the parameters are calculated from these, so we should not set them directly 
            even though we can. But this is up to the user, I'm not the boss of you.
        """
        self.mass = 1 # Mass of the meter, m = 1
        self.delta_E = self.Q_S*kB*self.T_S # Energy diff in the TLS, delta_E = Q_S*kB*T_S
        self.omega_meter = self.Q_M*kB*self.T_S/hbar # Angular frequency of the meter, omega = Q_M*kB*T_S/hbar
        self.T_m = self.x*self.T_S # Temperature of the meter, T_M = x*T_S 
        self.g = self.P*np.sqrt(kB*self.T_S/self.mass) # Coupling strength, g = P*sqrt(kB*T_S/m) but m = 1
        self.time = self.tau*2*np.pi/self.omega_meter # Interaction time, t = tau*2*pi/omega
        a = 1/( 1+np.exp( -self.delta_E/(kB*self.T_S) ) )
        b = np.exp(-self.delta_E/(kB*self.T_S))/( 1+np.exp( -self.delta_E/(kB*self.T_S) ) )
        self.tls_state = (a,b)
        if self.T_m == 0:
            self.beta = np.inf
        else:
            self.beta = 1/(kB*self.T_m) # Beta value





    
    #--------- Functions to update the hidden parameters of the system and meter ------------
    def update_total_levels(self):
        """
            Updates the total number of energy levels in the meter.
        """
        self.total_levels = int(10*np.ceil(kB*self.T_m/(hbar*self.omega_meter))+1)
        T_M = self.get_temp_meter()
        omega = self.get_omega()
        self.total_levels = int(10*np.ceil(kB*T_M/(hbar*omega))+1)
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
        return self.T_S
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
    def get_delta_E(self):
        """Getter function for the energy difference between the two states of the system.

        Returns:
            float: Energy difference between the two states of the system.
        """
        return self.delta_E
    def get_Q_S(self):
        """Getter function for the dimensionless scaling parameter for the TLS energy.

        Returns:
            float: Dimensionless scaling parameter for the TLS energy.
        """
        return self.Q_S
    def get_Q_M(self):
        """Getter function for the dimensionless scaling parameter for the meter energy.

        Returns:
            float: Dimensionless scaling parameter for the meter energy.
        """
        return self.Q_M
    def get_P(self):
        """Getter function for the dimensionless parameter that sets coupling strength and mass.

        Returns:
            float: Dimensionless parameter that sets coupling strength and mass.
        """
        return self.P
    def get_x(self):
        """Getter function for the normalized temperature of the meter.

        Returns:
            float: Normalized temperature of the meter.
        """
        return self.x
    def get_tau(self):
        """Getter function for the normalized time of interaction between the system and meter.

        Returns:
            float: Normalized time of interaction between the system and meter.
        """
        return self.tau
    # ----------------------------------------------------------------------------------------
    # --------------- Functions for testing the SystemAndMeter class --------------------------

