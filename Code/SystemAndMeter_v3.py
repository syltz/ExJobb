#########
# SystemAndMeter_v2.py
# This module defines the SystemAndMeter class which models a system and meter for quantum thermodynamics.
# It includes methods for calculating joint probabilities, conditional probabilities, entropies, and work extraction.
# It also includes methods for setting and getting the parameters of the system and meter.
# Just as a note, there are parameters called Q_S, Q_M, and P which are dimensionless scaling parameters for the system and meter.
# Q_S is the scaling parameter for the TLS energy, Q_M is the scaling parameter for the meter energy,
# and P is the scaling parameter for the coupling strength and mass.
# The system is a two-level system (TLS) and the meter is a harmonic oscillator.
# I recognize that the naming of the parameters is a bit confusing, but I made a poor choice at some point in the past
# and now I have to live with it.
#########


import numpy as np
import scipy as sp # Not currently used but can be uncommented if needed
from scipy.special import factorial, assoc_laguerre, gammaln

kB = 1
hbar = 1

class SystemAndMeter:
    def __init__(self, T_S=300, x=1, Q_S=1, Q_M=1, P=1, tau = 0.5, msmt_state=0, n_upper_limit=None, R=0, mass = 1):
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
            R (float): The dimensionless parameter that sets the dissipation rate of the meter, gamma = R*omega. Default is 0.

        """        
        self.T_S = T_S
        self.x = x
        self.Q_S = Q_S
        self.Q_M = Q_M
        self.P = P
        self.n = msmt_state
        self.tau = tau
        self.R = R
        self.mass = mass
        self.update_params()
        self.update_total_levels()
        if n_upper_limit == None:
            self.n_upper_limit = self.get_total_levels()
        else:
            self.n_upper_limit = n_upper_limit
        
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
        """
        Returns the Gibbs-Boltzmann probability at energy level(s) `n` for the meter's temperature.
        Fully vectorized: handles scalars or numpy arrays.

        Args:
            n (int or ndarray): The energy level(s) of the meter.

        Returns:
            float or ndarray: Probability or array of probabilities.
        """
        n = np.asarray(n)
        beta = self.beta
        omega = self.omega_meter
        T = self.T_m

        # Initialize output array
        p_n = np.zeros_like(n, dtype=np.float64)

        if T == 0:
            # Ground state only has nonzero population
            p_n[n == 0] = 1.0
            return p_n if p_n.size > 1 else float(p_n)
        
        Z_prime = 1 / (1 - np.exp(-hbar * omega / (kB * T)))
        p_n = np.exp(-beta * hbar * omega * n) / Z_prime

        return p_n if p_n.size > 1 else float(p_n)

    

    from scipy.special import gammaln, assoc_laguerre

    from scipy.special import gammaln, assoc_laguerre

    def shift_factor_matrix(self, n_vals, m_vals, t):
        """
        Computes |<n|D(t)|m>|^2 for all n, m combinations, safely using logs.
        """
        n_vals = np.atleast_1d(n_vals)
        m_vals = np.atleast_1d(m_vals)
        N, M = np.meshgrid(n_vals, m_vals, indexing='ij')  # Shape: (N, M)

        # Precompute alpha
        g = self.g
        omega = self.omega_meter
        mass = self.mass
        gamma = self.gamma
        R = self.R

        if omega * t < 1e-5:
            alpha = g * np.sqrt(mass * omega / (2 * hbar)) * t * (1 + 1j * R / 2)
        else:
            alpha = g * np.sqrt(mass / (2 * hbar * omega)) * (
                np.exp(-gamma * t / 2) * np.sin(omega * t)
                - 1j * (np.exp(-gamma * t / 2) * np.cos(omega * t) - 1)
            )

        abs_alpha_sq = np.abs(alpha) ** 2
        log_abs_alpha = np.log(np.abs(alpha) + 1e-300)  # avoid log(0)

        # Compute log(factorial ratios)
        log_fact_ratio = np.where(
            N >= M,
            gammaln(M + 1) - gammaln(N + 1),
            gammaln(N + 1) - gammaln(M + 1)
        )

        k = np.abs(N - M)
        laguerre_terms = np.where(
            N >= M,
            assoc_laguerre(abs_alpha_sq, M, k),
            assoc_laguerre(abs_alpha_sq, N, k)
        )
        laguerre_abs = np.abs(laguerre_terms) + 1e-300  # avoid log(0)
        log_laguerre = np.log(laguerre_abs)

        # log(|D|^2)
        log_D_sq = (
            -abs_alpha_sq
            + log_fact_ratio
            + 2 * k * log_abs_alpha
            + 2 * log_laguerre
        )

        D_sq = np.exp(log_D_sq)
        D_sq[~np.isfinite(D_sq)] = 0
        return D_sq



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

        g = self.g # Coupling strength
        omega = self.omega_meter # Angular frequency of the meter
        mass = self.mass # Mass of the meter
        gamma = self.gamma # Dissipation rate of the meter
        R = self.R # Dimensionless parameter that sets the dissipation rate of the meter
        # For numerical stability reasons, use Taylor expansion to first order for small values of omega*t 
        # Also, note that when R=gamma = 0 we recover the non-dissipative case.
        if omega*t < 1e-5: 
            alpha = g*np.sqrt(mass*omega/(2*hbar))*t*(1+1j*R/2)
        else:
            alpha = g*np.sqrt(mass/(2*hbar*omega))*(np.exp(-gamma*t/2)*np.sin(omega*t) -1j*(np.exp(-gamma*t/2)*np.cos(omega*t)-1))

        try:
            if n >= m:
                # Check if the factorial ratio is zero and if either the real or imaginary part
                # of alpha**(n-m) is inf or infj
                if self.factorial_ratio(m,n) == 0 and np.abs(alpha)>0 and (n-m) >= np.log(np.finfo(np.float64).max)/np.log(np.abs(alpha)):
                    d = 0
                else:
                    d = np.abs(np.exp(-np.abs(alpha)**2/2)*np.sqrt(self.factorial_ratio(m,n))*\
                               alpha**(n-m)*assoc_laguerre(np.abs(alpha)**2, m, np.abs(m-n)))**2
            else:
                if self.factorial_ratio(n,m) == 0 and np.abs(alpha)>0 and (m-n) >= np.log(np.finfo(np.float64).max)/np.log(np.abs(alpha)):
                    d = 0
                else:
                    d = np.abs(np.exp(-np.abs(alpha)**2/2)*np.sqrt(self.factorial_ratio(n,m))*\
                               (-np.conjugate(alpha))**(m-n)*assoc_laguerre(np.abs(alpha)**2, n, np.abs(m-n)))**2
            return d
        except RuntimeWarning:
            print(f"Warning: n = {n}, m = {m}, t = {t}, alpha = {alpha}")
            exit()
    
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
        if n is None:
            n = self.n
        if t is None:
            t = self.time

        n = np.atleast_1d(n)
        m_vals = np.arange(0, self.total_levels + 1)

        pop_m = self.population_distribution(m_vals)  # shape: (M,)
        p0_n = self.tls_state[0] * self.population_distribution(n)

        # Vectorized shift matrix
        shift_mat = self.shift_factor_matrix(n_vals=n, m_vals=m_vals, t=t)  # shape: (N, M)

        # Weighted sum across m
        p1_n = self.tls_state[1] * shift_mat @ pop_m  # shape: (N,)

        return p0_n, p1_n


        
    def conditional_probability(self, n=None, t=None):
        """Conditional probabilities of the system being in states 0 and 1 given meter state n at time t."""
        n = self.n if n is None else n
        t = self.time if t is None else t

        p0_n, p1_n = self.joint_probability(n=n, t=t)
        total = p0_n + p1_n

        if total == 0:
            return 0.0, 0.0  # Avoid division by zero

        p0_given = p0_n / total
        p1_given = p1_n / total
        return p0_given, p1_given



    def conditional_entropy(self, n=None, time=None):
        """Conditional entropy of the system given the meter is in state n at time t."""
        time = self.time if time is None else time
        n = self.n if n is None else n

        p0, p1 = self.conditional_probability(n=n, t=time)

        # Use a stable formulation: x*log(x) → 0 as x → 0
        terms = []
        for p in [p0, p1]:
            if p > 0:
                terms.append(p * np.log(p))
        S_n = -kB * np.sum(terms)
        return S_n

    
    def entropy(self, time=None):
        """Marginal entropy of the system at time t."""
        time = self.time if time is None else time
        S = 0.0

        for n in range(self.total_levels + 1):
            p0_n, p1_n = self.joint_probability(n, time)
            P_n = p0_n + p1_n
            if P_n > 0:
                S_n = self.conditional_entropy(n=n, time=time)
                S += P_n * S_n
        return S

    
    def mutual_information(self, time=None):
        """Mutual information between system and meter at time t."""
        time = self.time if time is None else time
        S_0 = self.entropy(time=0.0)
        S_t = self.entropy(time)
        return (S_0 - S_t)[0]

    
    def observer_information(self, time=None, n=None, n_upper_limit=None):
        """Calculates the observers information about the system after measurement in n at time t.
        N.B. This is not the same as the mutual information, rather it is the information gained by the observer after
        projective measurement. This also turns out to no be so useful, but the function is here anyway.

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
        if (n_upper_limit==None):
            n_upper_limit = self.n_upper_limit
        p_n = 0
        for m in range(n, n_upper_limit+1):
            p0_n, p1_n = self.joint_probability(n=m, t=time)
            p_n += (p0_n + p1_n)
        I_O = -kB * (p_n*np.log(p_n) + (1-p_n)*np.log(1-p_n))
        return I_O[0]

    def ergotropy(self, time=None):
        """
        Calculates the ergotropy of the system at a given time.
        """
        if time is None:
            time = self.time

        delta_E = self.delta_E
        n_primes = np.arange(1, self.n_upper_limit)

        p0_array, p1_array = self.joint_probability(n=n_primes, t=time)

        # Only keep terms where work can be extracted
        mask = p1_array > p0_array
        ergotropy = delta_E * np.sum(p1_array[mask] - p0_array[mask])
        if not np.isfinite(ergotropy):
            ergotropy = 0

        return ergotropy



    def work_extraction(self, time=None, work_type='ergotropy'):
        """Calculates the work extracted from the system at time t.


        Args:
            time (float, optional): System time to calculate the work extracted at. Defaults to self.time if not set.
            work_type (str, optional): The type of work extraction to calculate. Defaults to 'ergotropy'.

        Returns:
            float: The work extracted from the system at time t.
        """
        if (time==None):
            time = self.time
        if work_type == 'ergotropy':
            return self.ergotropy(time)
        elif work_type == 'excess':
            return self.work_extraction_excess(time)


    def work_extraction_excess(self, time=None):
        """Calculates the work extracted from the system at time t. To be clear, this is not used in the paper and it differs
        from the ergotropy. This is essentially the work extracted from the system when disregarding losses, essentially saying that
        from an initial thermal state a|0><0| + b|1><1| you could extract b*delta_E from the system, i.e. disregarding the losses
        due to absorption.

        Args:
            time (float, optional): System time to calculate the work extracted at. Defaults to self.time if not set.

        Returns:
            float: The work extracted from the system at time t. sum_n P(n,t)*(P(1|n,t) - P(1|n,0))*delta_E
        """
        if (time==None):
            time = self.time
        delta_E = self.delta_E
        a, b = self.tls_state
        W = 0
        lower_limit = self.n
        upper_limit = self.n_upper_limit
        for n in range(lower_limit, upper_limit+1):
            p0_n, p1_n = self.joint_probability(n=n, t=time)
            W+=a*p1_n - b*p0_n
        W *= delta_E
        return W

    def work_measurement(self, time=None):
        """Calculates the work required to measure the meter at time t.

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
        wt = 2*np.pi*self.tau # t = tau*2*pi/omega
        if wt/(2*np.pi) < 1e-5:
            return b*m*g**2*(wt)**2/2
        else:
            return b*m*g**2*(1-np.cos(wt))


    def quality_factor(self, time=None):
        """Calculates the quality factor of the measurement at time t. For some definition of a quality factor.
        Not sure how useful this is, but it was considered at some point in the past so it's here.

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
        """ I don't think I actually need this function but I'm keeping it here just in case.
            Probably don't use this without thorough testing."""
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

    def zeno_limit_work_measurement(self):
        """Calculates the work required to measure the meter in the Zeno limit.
        This turns out to not be needed, rather the normal W_meas is sufficient.

        Returns:
            float: The work required to measure the meter in the Zeno limit.
        """
        g = self.g
        m = self.mass
        b = self.tls_state[1]
        wt = 2*np.pi*self.tau 
        W_meas = b*m*g**2 * wt**2/2
        return W_meas
    def zeno_limit_work_extraction(self):
        """Calculates the work extracted from the system in the Zeno limit.
        Probably don't use this, there still might be something off about this, I'm not convinced it's correct.

        Returns:
            float: The work extracted from the system in the Zeno limit.
        """
        a, b = self.tls_state
        delta_E = self.delta_E
        g = self.g
        m = self.mass
        wt = 2*np.pi*self.tau # t = tau*2*pi/omega
        t = self.time
        beta = self.beta
        omega = self.omega_meter
        n_prime = self.n
        W = a*b*delta_E*(g**2*wt*t*m*hbar/2)*n_prime*(1-np.exp(beta*hbar*omega))**2*np.exp(-beta*hbar*omega*(n_prime+1))
        return W

    # Functions to set the parameters of the system and meter
    def set_n_upper_limit(self, n_upper_limit):
        """Setter function for the upper limit of the energy levels of the meter to measure.

        Args:
            n_upper_limit (int): Upper limit of the energy levels of the meter to measure.
        """
        if n_upper_limit == None:
            self.n_upper_limit = self.get_total_levels()
        else:
            self.n_upper_limit = n_upper_limit
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
        self.full_update()
    def set_R(self, R):
        """Setter function for the dimensionless parameter that sets the dissipation rate of the meter.

        Args:
            R (float): Dimensionless parameter that sets the dissipation rate of the meter.
        """
        self.R = R
        self.update_params()
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
    def set_gamma(self, gamma):
        """Setter function for the dissipation rate of the meter.

        Args:
            gamma (float): Dissipation rate of the meter.
        """
        self.gamma = gamma
    def set_mass(self, mass):
        """Setter function for the mass of the meter.

        Args:
            mass (float): Mass of the meter in meV/c^2.
        """
        self.mass = mass


    #--------- Functions to update the hidden parameters ofthe system and meter ------------
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
        self.gamma = self.R*self.omega_meter # Dissipation rate of the meter, gamma = R*omega
        a = 1/( 1+np.exp( -self.delta_E/(kB*self.T_S) ) ) # Initial ground state population of the TLS
        b = np.exp(-self.delta_E/(kB*self.T_S))/( 1+np.exp( -self.delta_E/(kB*self.T_S) ) ) # Initial excited state population of the TLS
        self.tls_state = (a,b)
        if self.T_m == 0:
            self.beta = np.inf
        else:
            self.beta = 1/(kB*self.T_m) # Inverse temperature of the meter, beta = 1/(kB*T_M)

    def update_total_levels(self):
        """
            Updates the total number of energy levels in the meter.
        """
        #self.total_levels = int(10*np.ceil(kB*self.T_m/(hbar*self.omega_meter))+1)
        T_M = self.get_temp_meter()
        omega = self.get_omega()
        self.total_levels = int(30*np.ceil(kB*T_M/(hbar*omega))+1)
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
    def get_n_upper_limit(self):
        """Getter function for the upper limit of the energy levels of the meter to measure.

        Returns:
            int: Upper limit of the energy levels of the meter to measure.
        """
        return self.n_upper_limit
    def get_gamma(self):
        """Getter function for the dissipation rate of the meter.

        Returns:
            float: Dissipation rate of the meter.
        """
        return self.gamma
    def get_R(self):
        """Getter function for the dimensionless parameter that sets the dissipation rate of the meter.

        Returns:
            float: Dimensionless parameter that sets the dissipation rate of the meter.
        """
        return self.R