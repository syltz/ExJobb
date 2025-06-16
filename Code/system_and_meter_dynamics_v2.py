import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import scipy as sp
from SystemAndMeter_v2 import SystemAndMeter
import pandas as pd

# Throughout this code, the meter is assumed to be a harmonic oscillator
# and the system is assumed to be a two-level system.
kB = 1
hbar = 1

def main():
    temp_system = 1.0
    x = 1.0
    Q_S = 1.0
    Q_M = 1.0
    P = 1.0
    msmt_state = 0
    sam = SystemAndMeter(T_S=temp_system, x=x, Q_S=Q_S, Q_M=Q_M, P=P, msmt_state=msmt_state)
    # Dictionaries of parameters to test. They are here to ensure consistency in the parameters.
    # Just uncomment the one you want to test and comment the others.
    # To be clear to anyone reading this later, most of these parameters have been generated
    # by running a simple optimizer on the old W_ext function. Input whatever parameters you want
    # to test.
    params_article = {'Q_S': 4.33, 'P': 1., 'Q_M': 1.5, 'x': 0.01, 'tau': 0.25,\
                      'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_article'} # Parameters from the article
    
    
    
    
    params = params_article
    sam = SystemAndMeter(T_S=temp_system, x=params['x'], Q_S=params['Q_S'], Q_M=params['Q_M'], P=params['P'], msmt_state=params['n_prime'])
    sam.set_tau(params['tau'])
    sam.set_n(params['n_prime'])
    sam.set_n_upper_limit(params['n_upper_limit'])
    sam.set_P(params['P'])
    sam.set_Q_S(params['Q_S'])
    sam.set_Q_M(params['Q_M'])
    sam.set_R(0.0)
    sam.set_x(params['x'])
    sam.full_update()
    
    temp_range = np.linspace(0.0, 1.0, 100)
    dE = sam.get_delta_E() # Energy difference between the two levels
    # Write header to the file
    with open('data/article_data/ergotropy_extraction.csv', 'w') as f:
        # First write the fixed parameters
        f.write(f'T_S = {temp_system}, Q_S = {params["Q_S"]}, Q_M = {params["Q_M"]}, P = {params["P"]}, tau = {params["tau"]}\n')
        f.write('Temperature, W_erg, W_ad\n')
    for x in temp_range:
        W_erg = 0
        W_ad = 0
        sam.set_x(x)
        sam.full_update()
        for n in range(sam.get_total_levels()):
            sam.set_n(n)
            p_0, p_1 = sam.joint_probability(n=n) # Joint probabilities
            p_0_cond, p_1_cond = sam.conditional_probability(n=n) # Conditional probabilities
            if p_1_cond > p_0_cond:
                W_erg += (p_1 + p_0)*(p_1_cond - p_0_cond) # Ergotropy
                # Calculate the passive temperature resulting after ergotropy extraction
                # Note that after population inversion p_1 -> p_0 and p_0 -> p_1 so in fact p_1 defines the ground state in this case
                T_p = -dE/(kB*np.log(p_0_cond/p_1_cond)) # Passive temperature
                if T_p == 0.0:
                    T_p = 1e-20 # Avoid division by zero
                W_ad += ( 0.5 - np.exp(-dE/(kB*T_p)) / (1+np.exp(-dE/(kB*T_p))) ) # Adiabatic work
        # Write the results to the file
        with open('data/article_data/ergotropy_extraction.csv', 'a') as f:
            f.write(f'{x}, {W_erg}, {W_ad}\n')
    print("Comparison done, saved to data/article_data/ergotropy_extraction.csv")
    exit()


    # Data for the meter level as function of temperature
    params_vs_temp(sam, temp_range=np.linspace(0.0, 1.0, 100), fname='data/article_data/params_vs_temp_test.csv', type='ergotropy')
    sam.set_x(params['x'])
    sam.full_update()
    meter_level = np.array([])
    temp_range = np.linspace(0.0, 2.0, 1000)
    for T in temp_range:
        sam.set_x(T)
        sam.full_update()
        p_0 = 1
        p_1 = 0
        n = 0

        while (p_1 < p_0 and n < sam.get_total_levels()):
            p_0, p_1 = sam.conditional_probability(n=n)
            if p_1 > p_0:
                break
            n += 1
        meter_level = np.append(meter_level, n)
    df = pd.DataFrame({'Temperature': temp_range, 'Meter Level': meter_level})
    df.to_csv('data/article_data/meter_level_vs_temp_test.csv', index=False)

    sam.set_x(params['x'])

    # ----------- Data for the work against time function in the appendix -----------
    tau_vals = np.linspace(0.0, 1.0, 1000)
    ratios = [0.01, 0.1, 1.0]
    for ratio in ratios:
    # g_eff^2 = P^2 *k_B * T_S and Delta E = k_B * T_S * Q_S
    # Thus we have g_eff^2/Delta E = P^2/Q_S = ratio
    # Therefore we can set P = np.sqrt(ratio * Q_S)
        sam.set_P(np.sqrt(ratio*sam.get_Q_S())) 
        sam.full_update()
        params_vs_time(sam, tau_range=tau_vals, fname=f'params_vs_time_ratio={ratio}.csv')
    # --------------------------------------------------------------------------------

    # ----------- Data for the heatmaps -----------
    # Reset all the parameters again to make sure we start from the same point
    sam.set_Q_S(params['Q_S'])
    sam.set_Q_M(params['Q_M'])
    sam.set_P(params['P'])
    sam.set_x(params['x'])
    sam.set_tau(params['tau'])
    sam.set_n(params['n_prime'])
    sam.set_n_upper_limit(params['n_upper_limit'])
    sam.set_R(0.0)
    sam.full_update()

    # Times at which to generate the heatmaps
    tau_vals = [1e-6, 0.125, 0.25, 0.5]
    for tau in tau_vals:
        sam.set_tau(tau)
        sam.full_update()
        # Heatmap for the ergotropy
        phase_boundary_multidata_coupling(sam, temp_range=np.linspace(0.0, 2.0, 1000),\
                                           fname=f'phase_boundary_multidata_tau={tau}_R=0.0.csv',\
                                              work_type='ergotropy')
    # --------------------------------------------------------------------------------

    # -------- Data for the thermodynamic and information efficiencies -----------
    # Reset all the parameters again to make sure we start from the same point
    sam.set_Q_S(params['Q_S'])
    sam.set_Q_M(params['Q_M'])
    sam.set_P(params['P'])
    sam.set_x(params['x'])
    sam.set_tau(params['tau'])
    sam.set_n(params['n_prime'])
    sam.set_n_upper_limit(params['n_upper_limit'])
    sam.set_R(0.0)
    sam.full_update()

    temp_range = np.linspace(0.0, 1.0, 1000)
    params_vs_temp(sam, temp_range=temp_range, fname='params_vs_temp.csv', type='ergotropy')


        

def positive_work_extraction(sam, fname=None, times=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]):
    """ Investigates whether positive work extraction is possible for the given system and meter.
        By varying time, temperature, and coupling strength we can determine calculate
        all the conditional probabilities p(1|n,t) and plot them.

        Args:
            sam (SystemAndMeter): The coupled system and meter object.
            fname (str, optional): The filename to save the plot. Saves to default location if None. Defaults to None.
            times (ndarray or list, optional): The times at which to evaluate the conditional probabilities. Defaults to [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0].
    """ 
    temps = np.linspace(0.0, 2, 10)
    # Keep the coupling strength fixed to 1.0
    sam.set_P(1.0)
    # Compute the conditional probabilities at t=0 for each temperature
    sam.set_tau(0.0)
    cond_probs_t0 = {}
    for T in temps:
        sam.set_x(T)
        cond_probs_t0[T] = np.zeros(sam.get_total_levels(), dtype=np.float64)
        for n in range(sam.get_total_levels()):
            sam.set_n(n)
            cond_probs_t0[T][n] = sam.conditional_probability(n, 0.0)[1]
    # Make plots for each time with the conditional probabilities
    fig, axs = plt.subplots(len(times), 1, figsize=(16, 16))
    axs = np.atleast_1d(axs)
    all_lines = []
    all_labels = []
    extra_ticks = []
    for i,t in enumerate(times):
        sam.set_tau(t) # Set the normalized time
        for T in temps:
            sam.set_x(T) # Set the normalized temperature
            cond_probs = np.zeros(sam.get_total_levels(), dtype=np.float64) # Initialize the conditional probabilities
            #cond_probs_t0 = np.zeros(sam.get_total_levels(), dtype=np.float64)
            for n in range(sam.get_total_levels()):
                sam.set_n(n) # Set the meter level
                cond_probs[n] = sam.conditional_probability(n, t)[1] # Select the probability of the meter being in state 1
            #    cond_probs_t0[n] = sam.conditional_probability(n, 0.0)[1]

            # Just some plotting stuff that allows me to have a single legend for the entire figure
            #line, = axs[i].plot(np.arange(0, sam.get_total_levels()), cond_probs-cond_probs_t0, label=f'T = {T:.2f}')
            diff = cond_probs-cond_probs_t0[T]
            line, = axs[i].plot(np.arange(0, sam.get_total_levels()), diff, label=f'T = {T:.2f}')
            if i == 0:
                all_lines.append(line)
                all_labels.append(r'$T_{M}$ = '+f'{T:.2f}'+r'$T_{S}$')

            #axs[i].plot(np.arange(0, sam.get_total_levels()), cond_probs, label=f'T = {T:.2f}')
            # Find the first meter level where positive work extraction is possible
            positive_n = np.argmax(diff > 0)
            if (diff[positive_n] > 0):
                axs[i].plot(positive_n, diff[positive_n], 'x', color=line.get_color())
                extra_ticks = np.unique(np.concatenate((extra_ticks, [positive_n])))
        
        # Plotting setup
        axs[i].set_xlabel('Meter Level, $n$')
        axs[i].set_ylabel(r'$P(1|n,t)-P(1|n,t=0)$')
        axs[i].set_title(r'Time $\tau$ ='+f' {t}')
        axs[i].hlines(sam.get_tls_state()[1], 0, sam.get_total_levels(), color='black', linestyle='--', label=r'P(1, t=0)')
        if i == 0:
            hline_legend = Line2D([0], [0], color='black', linestyle='--', label=r'$P(1, t=0)=b$')
            all_lines.append(hline_legend)
            all_labels.append(r'$P(1, t=0)=b$')

        # Combine existing ticks with new ticks up to the highest meter level where positive work extraction is possible
        current_ticks = axs[i].get_xticks()
        new_ticks = np.unique(np.concatenate((current_ticks, extra_ticks)))
        axs[i].set_xticks(new_ticks)

        # Fix the x limits to be more appropriate
        axs[i].set_xlim(-0.1, sam.get_total_levels()+0.1)

    # Calculate the number of rows and columns for the subplots
    num_plots = len(times)
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots/num_cols))

    # Reorganize the subplots into appropriate rows and columns
    fig.tight_layout()#
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig)
    for i, ax in enumerate(axs):
        row = i // num_cols
        col = i % num_cols
        ax.set_position(gs[row, col].get_position(fig))
        ax.set_subplotspec(gs[row, col])

    # Add a single legend for the entire figure
    fig.legend(all_lines, all_labels)

    # Add a supertitle to the entire figure
    fig.suptitle(r'Cond. prob. $P(1|n,t)$ for different $T_{M}$ at different times with system temp $T_{S}$ ='+f' {sam.get_temp_system()}K. Ticks mark where $P(1|n,t)-P(1|n,t=0) > 0$')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust the layout to make room for the supertitle

    if fname is None:
        plt.savefig(f'../images/positive_work_extraction_QM={sam.get_Q_M()}_QS={sam.get_Q_S()}.png')
    else:
        plt.savefig(fname)

def first_positive_W_ext(sam):
    """ Find the first level n where positive work extraction is possible at whatever time and temperature
        the system and meter are currently at.
        
        Args:
   #e3*sp.constants.physical_constants['Boltzmann constant in eV/K'][0] # Boltzmann constant in meV/K         sam (SystemAndMeter): The coupled system and meter object.
        
        Returns:
            int: The first meter level where positive work extraction is possible."""
    cond_probs_t0 = np.zeros(sam.get_total_levels(), dtype=np.float64)

    cond_probs = np.zeros(sam.get_total_levels(), dtype=np.float64)
    for n in range(sam.get_total_levels()):
        sam.set_n(n)
        cond_probs[n] = sam.conditional_probability()[1]
        cond_probs_t0[n] = sam.conditional_probability(n=n, t=0.0)[1]
    diff = cond_probs-cond_probs_t0
    positive_n = np.argmax(diff > 0)
    return positive_n

def work_minimizer(x, sam):
    """ Calculate the net work done on the system by the meter.

    Args:
        sam (SystemAndMeter): The coupled system and meter object.

    Returns:
        float: The net work done on the system by the meter."""
    Q_S, P, Q_M, T_M, tau = x
    sam.set_Q_S(Q_S)
    sam.set_P(P)
    sam.set_Q_M(Q_M)#
    sam.set_x(T_M)
    sam.set_tau(tau)
    sam.full_update()
    W_ext = sam.work_extraction()
    W_meas = sam.work_measurement()
    return -(W_ext - W_meas)

def find_pos_net_work(sam, n=1):
    """ Find the maximum net work extraction possible for the given system and meter.

    Args:
        sam (SystemAndMeter): The coupled system and meter object.

    Returns:
        float: The maximum net work extraction possible."""
    from scipy.optimize import minimize
    # Set the initial guess for the minimizer and bounds that are non-zero and positive
    x0 = [sam.get_Q_S(), sam.get_P(), sam.get_Q_M(), sam.get_x(), sam.get_tau()]
    sam.set_n(n)
    # Add some small random noise to the initial guess to avoid getting stuck in local minima
    x0 = [x + np.random.normal(-0.1, 0.1) for x in x0]
    res = minimize(work_minimizer, x0, args=(sam), bounds=[(1e-2, None), (1e-2, None), (0.2, None), (1e-2, None), (1e-2,1)], method='L-BFGS-B')
    return -res.fun, res.x

def work_minimizer_fixed_temps(x, sam: SystemAndMeter):
    """ Calculate the net work done on the system by the meter with fixed temperatures.

    Args:
        sam (SystemAndMeter): The coupled system and meter object.

    Returns:
        float: The net work done on the system by the meter.
        list: The system parameters."""
    Q_S, P, Q_M, tau = x
    sam.set_Q_S(Q_S)
    sam.set_P(P)
    sam.set_Q_M(Q_M)
    sam.set_tau(tau)
    sam.full_update()
    W_ext = sam.ergotropy()
    W_meas = sam.work_measurement()
    return -(W_ext - W_meas)
def find_pos_net_work_fixed_temps(sam: SystemAndMeter, n=1, T=1.0):
    from scipy.optimize import minimize
    # Set the initial guess for the minimizer and bounds that are non-zero and positive
    sam.set_x(T)
    sam.set_n(n)
    P_scale = 1e8
    x0 = [np.random.uniform(0.1, 10), np.random.uniform(0.1, 10), np.random.uniform(0.1, 10), np.random.uniform(0.1, 0.5)]
    # Add some small random noise to the initial guess to avoid getting stuck in local minima
    res = minimize(work_minimizer_fixed_temps, x0, args=(sam), bounds=[(1e-2, None), (0.1, None), (1e-2, None), (1e-1, 0.5)], method='L-BFGS-B')
    return -res.fun, res.x

def params_vs_temp(sam: SystemAndMeter, temp_range=np.linspace(0.0, 2.0, 100), fname="data/params_vs_temp.csv", fixed=None, n_upper_limit=None, type='ergotropy'):
    """ Investigate how the various system parameters vary with temperature.
        The parameters are the work W, the system heat Q_S, the meter heat Q_M, the information I=I_m+I_obs
        Saves the data to a csv file.
        
        Args:
            sam (SystemAndMeter): The coupled system and meter object.
            temp_range (ndarray or list, optional): The temperature range to evaluate the parameters at. Defaults to np.linspace(0.0, 2.0, 100).
            fname (str, optional): The filename to save the data to. Defaults to "data/params_vs_temp.csv".
    """
    results = {
        'Temperature': [],
        "Work": [],
        "System Heat": [],
        "Meter Heat": [],
        "Observer Information": [],
        "Mutual Information": [],
        "Information": []
    }
    for T in temp_range:
        sam.set_x(T)
        sam.full_update()
        if fixed is None:
            n = first_positive_W_ext(sam)
            sam.set_n(n)
        else:
            sam.set_n(fixed)
        if n_upper_limit is not None:
            sam.set_n_upper_limit(n_upper_limit)
        else:
            sam.set_n_upper_limit(sam.get_total_levels())
        # Check if we're in the Zeno regime
        if type == 'ergotropy':
            W_ext = sam.ergotropy()
        elif type == 'excess work':
            W_ext = sam.work_extraction_excess()
        W_meas = sam.work_measurement()
        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas
        I_obs = sam.observer_information()
        I_m = sam.mutual_information()
        I = I_obs + I_m
        results['Temperature'].append(T)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)
    df = pd.DataFrame(results)

    # Write the header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f},\
                    Coupling strength: {sam.get_P():.3f}, \
                         Delta_E: {sam.get_Q_S():.3f}, Omega: {sam.get_Q_M():.3f}, \
                            Period: {sam.get_tau():.3f}\n")
    # Append the data to the file
    df.to_csv(fname, mode='a', index=False)
    print(f"Parameters vs temperature saved to {fname}")

def params_vs_omega_per_delta_E(sam: SystemAndMeter, omega_range=np.linspace(0.0, 2.0, 100), fname="data/params_vs_omega_per_delta_E.csv", fixed=None):
    """ Investigate how the various system parameters vary with the ratio of the meter frequency to the system energy splitting.
        The parameters are the work W, the system heat Q_S, the meter heat Q_M, the information I=I_m+I_obs
        Saves the data to a csv file.
        
        Args:
            sam (SystemAndMeter): The coupled system and meter object.
            omega_range (ndarray or list, optional): The omega/delta_E range to evaluate the parameters at. Defaults to np.linspace(0.0, 2.0, 100).
            fname (str, optional): The filename to save the data to. Defaults to "data/params_vs_omega_per_delta_E.csv".
    """
    results = {
        'hw/dE': [],
        "Work": [],
        "System Heat": [],
        "Meter Heat": [],
        "Observer Information": [],
        "Mutual Information": [],
        "Information": []
    }
    for C_M in omega_range:
        temp_Q_S = sam.get_Q_S()
        sam.set_Q_S(1)
        sam.set_Q_M(C_M)
        sam.full_update()
        if fixed is None:
            n = first_positive_W_ext(sam)
        else:
            n = fixed
        sam.set_n(n)
        sam.set_n_upper_limit(sam.get_total_levels())
        hw_per_delta_E = C_M 
        # Check if we're in the Zeno regime
        if sam.get_tau() < 1e-5:
            W_ext = sam.zeno_limit_work_extraction()
            W_meas = sam.zeno_limit_work_measurement()
        else:
            W_ext = sam.work_extraction()
            W_meas = sam.work_measurement()
        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas
        I_obs = sam.observer_information()
        I_m = sam.mutual_information()
        I = I_obs + I_m
        results['hw/dE'].append(hw_per_delta_E)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)
    df = pd.DataFrame(results)
    sam.set_Q_S(temp_Q_S)
    # Write the header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f},\
                    Coupling strength: {sam.get_P():.3f}, \
                        Period: {sam.get_tau():.3f}, \
                            Temperature: {sam.get_x():.3f}\n")
    # Append the data to the file
    df.to_csv(fname, mode='a', index=False)
    print(f"Parameters vs omega/delta_E saved to {fname}")

def params_vs_time(sam: SystemAndMeter, tau_range=np.linspace(0.0, 2.0, 100), fname="data/params_vs_time.csv", fixed=None, type='ergotropy'):
    """ Investigate how the various system parameters vary with time.
        The parameters are the work W, the system heat Q_S, the meter heat Q_M, the information I=I_m+I_obs
        Saves the data to a csv file.
        
        Args:
            sam (SystemAndMeter): The coupled system and meter object.
            tau_range (ndarray or list, optional): The time range to evaluate the parameters at. Defaults to np.linspace(0.0, 2.0, 100).
            fname (str, optional): The filename to save the data to. Defaults to "data/params_vs_time.csv".
    """
    results = {
        'Time': [],
        "Work": [],
        "System Heat": [],
        "Meter Heat": [],
        "Observer Information": [],
        "Mutual Information": [],
        "Information": []
    }
    for tau in tau_range:
        sam.set_tau(tau)
        sam.full_update()
        if fixed is None:
            n = first_positive_W_ext(sam)
        else:
            n = fixed
        sam.set_n(n)
        sam.set_n_upper_limit(sam.get_total_levels())
        # Check if we're in the Zeno regime, actually probably don't use this.
        # The Zeno regime function might not be correct.
        if type == 'ergotropy':
            W_ext = sam.ergotropy()
        elif type == 'excess work':
            W_ext = sam.work_extraction_excess()
        W_meas = sam.work_measurement()
        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas
        I_obs = sam.observer_information()
        I_m = sam.mutual_information()
        I = I_obs + I_m
        results['Time'].append(tau)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)
    df = pd.DataFrame(results)

    # Write the header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f}, \
                    Coupling strength: {sam.get_P():.3f}, \
                        Delta_E: {sam.get_Q_S():.3f}, Omega: {sam.get_Q_M():.3f}, \
                            Temperature: {sam.get_x():.3f}\n")
    # Append the data to the file
    df.to_csv(fname, mode='a', index=False)
    print(f"Parameters vs time saved to {fname}")

def params_vs_coupling(sam: SystemAndMeter, g_range=np.linspace(0.0, 2.0, 100), fname="data/params_vs_coupling.csv", fixed=None):
    """ Investigate how the various system parameters vary with the coupling strength.
        The parameters are the work W, the system heat Q_S, the meter heat Q_M, the information I=I_m+I_obs
        Saves the data to a csv file.
        
        Args:
            sam (SystemAndMeter): The coupled system and meter object.
            g_range (ndarray or list, optional): The coupling strength range to evaluate the parameters at. Defaults to np.linspace(0.0, 2.0, 100).
            fname (str, optional): The filename to save the data to. Defaults to "data/params_vs_coupling.csv".
    """
    results = {
        'Coupling Strength': [],
        "Work": [],
        "System Heat": [],
        "Meter Heat": [],
        "Observer Information": [],
        "Mutual Information": [],
        "Information": []
    }
    for g in g_range:
        sam.set_P(g)
        sam.full_update()

        if fixed is None:
            n = first_positive_W_ext(sam)
        else:
            n = fixed
        sam.set_n(n)
        sam.set_n_upper_limit(sam.get_total_levels())
        # Check if we're in the Zeno regime
        if sam.get_tau() < 1e-5:
            W_ext = sam.zeno_limit_work_extraction()
            W_meas = sam.zeno_limit_work_measurement()
        else:
            W_ext = sam.work_extraction()
            W_meas = sam.work_measurement()
        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas
        I_obs = sam.observer_information()
        I_m = sam.mutual_information()
        I = I_obs + I_m
        results['Coupling Strength'].append(g)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)

    df = pd.DataFrame(results)
    #Write the header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f},\
                    Delta_E: {sam.get_Q_S():.3f}, Omega: {sam.get_Q_M():.3f}, \
                        Period: {sam.get_tau():.3f}, \
                            Temperature: {sam.get_x():.3f}\n")
    # Append the data to the file
    df.to_csv(fname, mode='a', index=False)
    print(f"Parameters vs coupling saved to {fname}")


def zeno_cross_over(sam: SystemAndMeter, temp_range=np.linspace(0.0, 2.0, 100), fname="data/zeno_cross_over.csv", fixed = None):
    """ Investigate the positive net work condition for the Zeno limit as a function of temperature ratio.
    Assumes a fixed system temperature, coupling strength, period, omega, and delta_E.
    
    Args:
        sam (SystemAndMeter): The coupled system and meter object.
        temp_range (ndarray or list, optional): The temperature range to evaluate the Zeno cross over at. Defaults to np.linspace(0.0, 2.0, 100).
        fname (str, optional): The filename to save the data to. Defaults to "data/zeno_cross_over.csv".
        fixed (int, optional): The meter level to fix the system to. Defaults to None.
    """
    if fixed == None:
        n = first_positive_W_ext(sam)
    else:
        n = fixed
    sam.set_n(n) # Set the meter level to measure from
    hw = hbar*sam.get_omega() # Constant hbar*omega, the level spacing of the meter
    C = 2*hw/(sam.get_tls_state()[0]*sam.get_delta_E()) # C = 2*hw/(a*delta_E), a constant that appears in the Zeno limit
    results = {
        'Temperature Ratio': [],
        "Work condition": [],
        "Constant limit": []
    }
    for x in temp_range:
        sam.set_x(x) # Set the temperature ratio
        sam.full_update() # Update the system and meter
        beta = sam.get_beta() # The inverse temperature of the system
        # Calculate the condition for positive net work in the Zeno limit
        exp_term = np.exp(-beta*hw*(n+1))
        with np.errstate(over='ignore'):
            cosh_term = np.cosh(beta*hw)
        if exp_term == 0 and np.isinf(cosh_term):
            cond=0
        elif exp_term != 0 and np.isinf(cosh_term):
            print("We have a problem")
        else:
            cond = exp_term *( 1 +2*cosh_term*(n*np.exp(beta*hw) + 1/(1-np.exp(-beta*hw))) )
        # Append the results to the dictionary
        results['Temperature Ratio'].append(x)
        results['Work condition'].append(cond)
        results['Constant limit'].append(C)
    df = pd.DataFrame(results)
    # Write the header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f},\
                    Coupling strength: {sam.get_P():.3f}, \
                        Delta_E: {sam.get_Q_S():.3f}, Omega: {sam.get_Q_M():.3f}, \
                            Period: {sam.get_tau():.3f}\n")
    # Append the data to the file
    df.to_csv(fname, mode='a', index=False)
    print(f"Zeno cross over saved to {fname}") 


def params_vs_nprime(sam: SystemAndMeter, nprime_range=np.arange(0,10), fname='data/params_vs_nprime.csv'):
    """ Investigate how the various system parameters vary with the meter level nprime.
        The parameters are the work W, the system heat Q_S, the meter heat Q_M, the information I=I_m+I_obs
        Saves the data to a csv file.
        
        Args:
            sam (SystemAndMeter): The coupled system and meter object.
            nprime_range (ndarray or list, optional): The meter level range to evaluate the parameters at. Defaults to np.arange(0,10).
            fname (str, optional): The filename to save the data to. Defaults to "data/params_vs_nprime.csv".
    """
    results = {
        'Meter Level': [],
        "Work": [],
        "System Heat": [],
        "Meter Heat": [],
        "Observer Information": [],
        "Mutual Information": [],
        "Information": []
    }
    for n in nprime_range:
        sam.set_n(n)
        sam.set_n_upper_limit(sam.get_total_levels())
        # Check if we're in the Zeno regime
        if sam.get_tau() < 1e-5:
            W_ext = sam.zeno_limit_work_extraction()
            W_meas = sam.zeno_limit_work_measurement()
        else:
            W_ext = sam.work_extraction()
            W_meas = sam.work_measurement()
        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas
        I_obs = sam.observer_information()
        I_m = sam.mutual_information()
        I = I_obs + I_m
        results['Meter Level'].append(n)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)
    df = pd.DataFrame(results)
    # Write the header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f},\
                    Coupling strength: {sam.get_P():.3f}, \
                        Delta_E: {sam.get_Q_S():.3f}, Omega: {sam.get_Q_M():.3f}, \
                            Period: {sam.get_tau():.3f}, \
                                Temperature: {sam.get_x():.3f}\n")
    # Append the data to the file
    df.to_csv(fname, mode='a', index=False)
    print(f"Parameters vs nprime saved to {fname}")


def find_phase_boundary(sam: SystemAndMeter, temp_range=np.linspace(0.0, 2.0, 100), fname="data/phase_boundary.csv", work_type='ergotropy'):
    """ Investigate the phase boundary between the positive and negative net work regions.
        Saves the data to a csv file.
        
        Args:
            sam (SystemAndMeter): The coupled system and meter object.
            temp_range (ndarray or list, optional): The temperature range to evaluate the phase boundary at. Defaults to np.linspace(0.0, 2.0, 100).
            fname (str, optional): The filename to save the data to. Defaults to "data/phase_boundary.csv".
    """
    Q_S = sam.get_Q_S()
    range_1 = np.linspace(1e-5*Q_S, 0.1*Q_S, 300)
    range_2 = np.linspace(0.98*Q_S, 1*Q_S, 100)
    Q_M_range = np.concatenate((range_1, range_2))
    # Write the header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f},\
                    Coupling strength: {sam.get_P():.3f}, \
                            Period: {sam.get_tau():.3f}\n")
    # For each temperature, calculate the net work for each Q_M in the range
    # Find where the net work changes sign and save the results
    for T in temp_range:
        sam.set_x(T)
        sam.full_update()
        W_vals = np.zeros_like(Q_M_range)
        # Calculate the net work for each Q_M in the range
        for i, Q_M in enumerate(Q_M_range):
            sam.set_Q_M(Q_M)
            sam.full_update()
            W_vals[i] = sam.work_extraction(work_type=work_type) -sam.work_measurement()
        # Find where the net work changes sign, i.e. the phase boundary
        sign_changes = np.where(np.diff(np.sign(W_vals)))[0]
        hw_dE_values = Q_M_range[sign_changes] / Q_S

        # Append the data to the file
        with open(fname, mode="a") as file:
            file.write(f"{T},{hw_dE_values}\n")
    print(f"Phase boundary saved to {fname}")

def phase_boundary_multidata(sam: SystemAndMeter, temp_range=np.linspace(0.0, 2.0, 100), fname="data/testing.csv", work_type="ergotropy"):
    """ Investigate the phase boundary between the positive and negative net work regions.
        Saves the data to a csv file.
        
        Args:
            sam (SystemAndMeter): The coupled system and meter object.
            temp_range (ndarray or list, optional): The temperature range to evaluate the phase boundary at. Defaults to np.linspace(0.0, 2.0, 100).
            fname (str, optional): The filename to save the data to. Defaults to "data/phase_boundary.csv".
    """
    Q_S = sam.get_Q_S()
    Q_M_range = np.linspace(0.01, 10*Q_S, 500)
    # Write the header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f},\
                    Coupling strength: {sam.get_P():.3f}, \
                            Period: {sam.get_tau():.3f}\n")
        for T in temp_range:
            file.write(f"{T},")
        file.write("\n")

    # For each temperature, calculate the net work for each Q_M in the range
    # Find where the net work changes sign and save the results
    for Q_M in Q_M_range:
        sam.set_Q_M(Q_M)
        sam.full_update()
        # Calculate the net work for each Q_M in the range
        with open(fname, mode="a") as file:
            for i, T in enumerate(temp_range):
                sam.set_x(T)
                sam.full_update()
                W_ext = sam.work_extraction(work_type=work_type)
                W_meas = sam.work_measurement()
                W = W_ext - W_meas
                element = [Q_M/Q_S, W, W_ext, W_meas]
                file.write(f"{element},")

        # Append the data to the file
        with open(fname, mode="a") as file:
            file.write(f"\n")
    print(f"Phase boundary saved to {fname}")

def phase_boundary_multidata_coupling(sam: SystemAndMeter, temp_range=np.linspace(1e-2, 2.0, 100), fname="data/testing.csv", work_type="ergotropy"):
    """ Investigate the phase boundary between negative net work regions by varying the parameter
    g_eff**2 / delta E or in other words the ratio P^2/Q_S 

    Args:
        sam (SystemAndMeter): The coupled system and meter object.
        temp_range (ndarray, list, optional): The temperature range to investigate.
        fname (str, optional): the filename to save the data to. Defaults to "data/phase_boundary_coupling.csv".

    """
    Q_S = sam.get_Q_S()
    P_range = np.linspace(0.1*Q_S, 10*Q_S, 500)
    # Write header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f}, \
                Omega param: {sam.get_Q_M():.3f},\
                Period: {sam.get_tau():.3F}\n")
        for T in temp_range:
            file.write(f"{T},")
        file.write("\n")

    # For each temperature calculate the net work for each P in the range
    for P in P_range:
        sam.set_P(P)
        sam.full_update()
        # Calculate the net work for each Q_M in the range
        with open(fname, mode="a") as file:
            for i,T in enumerate(temp_range):
                sam.set_x(T)
                sam.full_update()
                W_ext = sam.work_extraction(work_type=work_type)
                W_meas = sam.work_measurement()
                W = W_ext - W_meas
                element = [P**2/Q_S, W, W_ext, W_meas]
                file.write(f"{element},")
        # New line for the next round of values
        with open(fname, mode="a") as file:
            file.write(f"\n")
    print(f"Phase boundary multidata saved to {fname}")

def probabilities_against_meter_level(sam: SystemAndMeter, fname="data/probabilities_against_meter_level.csv"):
    """ Creates a DataFrame with four columns where the row indices correspond to the meter level.
        The four columns are: a (the initial population of the TLS ground state), b (the initial population of the TLS excited state),
        p0 (P(0|n,t)), and p1 (P(1|n,t)). Here the n in the conditional probabilities is the meter level corresponding to the row index.
        Saves the DataFrame to a csv file.
        
        Args:
            sam (SystemAndMeter): The coupled system and meter object.
            fname (str, optional): The filename to save the data to. Defaults to "data/probabilities_against_meter_level.csv".
    """
    df = pd.DataFrame(columns=['a', 'b', 'p0', 'p1'])
    start_n = 0
    stop_n = sam.get_total_levels()
    a, b = sam.get_tls_state()
    # Preallocate the DataFrame
    df = pd.DataFrame(index=np.arange(start_n, stop_n), columns=['a', 'b', 'p0', 'p1'])
    df['a'] = a
    df['b'] = b
    for n in range(start_n, stop_n):
        sam.set_n(n)
        p0, p1 = sam.conditional_probability()
        df.at[n, 'p0'] = p0
        df.at[n, 'p1'] = p1
    df.to_csv(fname, index_label='Meter Level')

    print(f"Probabilities against meter level saved to {fname}")

def per_cycle_work(sam: SystemAndMeter, fname="data/per_cycle_work.csv", continuous_comparison=False):
    """ Calculates the work done per cycle for the system and meter. Also allows for comparison with a continuous measurement approximation.
        Saves the data to a csv file.
        
        Args:
            sam (list, SystemAndMeter): The coupled system and meter object. Accepts a list of SystemAndMeter objects.
            This is useful for parallelizing the calculation of work done per cycle.
            If a list is provided, the function will iterate over each SystemAndMeter object in the list.
            If a single SystemAndMeter object is provided, it will be used directly.
            fname (str, optional): The filename to save the data to. Defaults to "data/per_cycle_work.csv".
            continuous_comparison (bool, optional): Whether to include a continuous approximation of work done per cycle. Defaults to False.
    """
    tau_cont = 1e-9
    work_def = 'excess'
    def calc_cont_approx(sam):
        """ Helper function to calculate the continuous approximation of work done per cycle."""
        original_tau = sam.get_tau()
        sam.set_tau(tau_cont)
        sam.full_update()
        W_ext = sam.work_extraction()
        W_meas = sam.work_measurement()
        W = W_ext - W_meas
        I_obs = sam.observer_information()
        I_m = sam.mutual_information()
        I = I_obs + I_m
        sam.set_tau(original_tau)
        return W, W_ext, W_meas, I_obs, I_m, I

    # Write headers for the CSV file
    with open(fname, mode="w") as file:
        file.write("Temp_System,P,Q_S,Q_M,Tau,W,Q_S,Q_M,I_obs,I_m,I\n")

    if isinstance(sam, list):
        # If a list of SystemAndMeter objects is provided, iterate over each object
        for s in sam:
            W_ext = s.work_extraction(work_type=work_def)
            W_meas = s.work_measurement()
            W = W_ext - W_meas
            I_obs = s.observer_information()
            I_m = s.mutual_information()
            I = I_obs + I_m
            if continuous_comparison:
                W_cont, W_ext_cont, W_meas_cont, I_obs_cont, I_m_cont, I_cont = calc_cont_approx(s)
                # Append the results to the CSV file
                with open(fname, mode="a") as file:
                    file.write(f"{s.get_temp_system()},{s.get_P()},{s.get_Q_S()},{s.get_Q_M()},{s.get_tau()},{W},{W_ext},{W_meas},{I_obs},{I_m},{I}\n")
                    file.write(f"{s.get_temp_system()},{s.get_P()},{s.get_Q_S()},{s.get_Q_M()},{tau_cont},{W_cont},{W_ext_cont},{W_meas_cont},{I_obs_cont},{I_m_cont},{I_cont}\n")
            else:
                # Append the results to the CSV file
                with open(fname, mode="a") as file:
                    file.write(f"{s.get_temp_system()},{s.get_P()},{s.get_Q_S()},{s.get_Q_M()},{s.get_tau()},{W},{W_ext},{W_meas},{I_obs},{I_m},{I}\n")
    else:
        # If a single SystemAndMeter object is provided, use it directly
        W_ext = sam.work_extraction()
        W_meas = sam.work_measurement()
        W = W_ext - W_meas
        I_obs = sam.observer_information()
        I_m = sam.mutual_information()
        I = I_obs + I_m
        if continuous_comparison:
            W_cont, W_ext_cont, W_meas_cont, I_obs_cont, I_m_cont, I_cont = calc_cont_approx(sam)
            # Append the results to the CSV file
            with open(fname, mode="a") as file:
                file.write(f"{sam.get_temp_system()},{sam.get_P()},{sam.get_Q_S()},{sam.get_Q_M()},{sam.get_tau()},{W},{W_ext},{W_meas},{I_obs},{I_m},{I}\n")
                file.write(f"{sam.get_temp_system()},{sam.get_P()},{sam.get_Q_S()},{sam.get_Q_M()},{tau_cont},{W_cont},{W_ext_cont},{W_meas_cont},{I_obs_cont},{I_m_cont},{I_cont}\n")
        else:
            # Append the results to the CSV file
            with open(fname, mode="a") as file:
                file.write(f"{sam.get_temp_system()},{sam.get_P()},{sam.get_Q_S()},{sam.get_Q_M()},{sam.get_tau()},{W},{W_ext},{W_meas},{I_obs},{I_m},{I}\n")

    print(f"Per cycle work saved to {fname}")

def fixed_time_work(sam:SystemAndMeter, time_interval=1.0, fname="data/fixed_time_work.csv"):
    """Calculates the net work output for a fixed time interval. Will run the largest integer number of cycles
    possible in the time interval. 
    The results are saved to a CSV file with the following columns:

    Args:
        sam (SystemAndMeter, list): The coupled system and meter object or a list of SystemAndMeter objects.
            The function will iterate over each SystemAndMeter object in the list.
            If a single SystemAndMeter object is provided, it will be used directly.
        time_interval (float, optional): The fixed time interval for the calculation in units of the oscillator period.
            Defaults to 1.0.
        fname (str, optional): The filename used to save the data. Defaults to "data/fixed_time_work.csv".
    """
    work_def = 'excess'
    # Write the header lines manually
    with open(fname, mode="w") as file:
        file.write("Temp_System,P,Q_S,Q_M,Tau,W_ext,W_meas,Work\n")

    if isinstance(sam, list):
        # If a list of SystemAndMeter objects is provided, iterate over each object
        for s in sam:
            n = int(time_interval/s.get_tau())
            W_ext = 0
            W_meas = 0
            W = 0
            for i in range(n):
                W_ext += s.work_extraction(work_type=work_def)
                W_meas += s.work_measurement()
                W += W_ext - W_meas
            # Append the results to the CSV file
            with open(fname, mode="a") as file:
                file.write(f"{s.get_temp_system()},{s.get_P()},{s.get_Q_S()},{s.get_Q_M()},{s.get_tau()},{W_ext},{W_meas},{W}\n")
    else:
        # If a single SystemAndMeter object is provided, use it directly
        n = int(time_interval/sam.get_tau())
        W_ext = 0
        W_meas = 0
        W = 0 
        for i in range(n):
            W_ext += sam.work_extraction(work_type=work_def)
            W_meas += sam.work_measurement()
            W += W_ext - W_meas
        # Append the results to the CSV file
        with open(fname, mode="a") as file:
            file.write(f"{sam.get_temp_system()},{sam.get_P()},{sam.get_Q_S()},{sam.get_Q_M()},{sam.get_tau()},{W_ext},{W_meas},{W}\n")

    print(f"Fixed time work saved to {fname}")
    


if __name__=='__main__':
    main()

