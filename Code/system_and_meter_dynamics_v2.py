import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from matplotlib.lines import Line2D
import scipy as sp
from SystemAndMeter_v2 import SystemAndMeter
import pandas as pd
import warnings
warnings.filterwarnings("error")

# Throughout this code, the meter is assumed to be a harmonic oscillator
# and the system is assumed to be a two-level system.
# I also use mass = 1, hbar = 1, and kB = 1 for simplicity. I do however keep them
# in the equations for clarity and so that they can be easily changed if needed.
# It's mostly to avoid any potential numerical issues that may arise from the
# small values of hbar and kB.
kB = 1e3*sp.constants.physical_constants['Boltzmann constant in eV/K'][0] # Boltzmann constant in meV/K
hbar = 1e3*sp.constants.physical_constants['reduced Planck constant in eV s'][0]# Reduced Planck constant in meV s

def main():
    temp_system = 300.0
    x = 1.0
    Q_S = 1.0
    Q_M = 1.0
    P = 1.0
    msmt_state = 0
    sam = SystemAndMeter(T_S=temp_system, x=x, Q_S=Q_S, Q_M=Q_M, P=P, msmt_state=msmt_state)
    # Dictionaries of parameters to test. They are here to ensure consistency in the parameters.
    # Just uncomment the one you want to test and comment the others.
    params_opt = {'Q_S': 2.25, 'P': 0.73, 'Q_M': 0.2, 'x': 0.01, 'tau': 0.31,\
                   'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_opt'} # Optimal parameters
    params_naive = {'Q_S': 1.0, 'P': 1.0, 'Q_M': 1.0, 'x': 1.0, 'tau': 0.5,\
                   'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_naive'} # Naive parameters
    params_zeno = {'Q_S': 2.25, 'P': 0.72, 'Q_M': 0.2, 'x': 0.01, 'tau': 1e-9,\
                   'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_zeno'} # Zeno parameters
    params_opt_eq_temp = {'Q_S': 4.33, 'P': 1.04, 'Q_M': 1.51, 'x': 1.0, 'tau': 0.25,\
                   'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_opt_eq_temp'} # Optimal parameters but with equal temperatures
    params_opt_uneq_temp = {'Q_S': 4.33, 'P': 1.04, 'Q_M': 1.51, 'x': 0.01, 'tau': 0.25,\
                   'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_opt_uneq_temp'} # The above parameters but with unequal temperatures
    params_zeno_eq_temp = {'Q_S': 4.33, 'P': 1.04, 'Q_M': 1.51, 'x': 1.0, 'tau': 1e-9,\
                   'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_zeno_eq_temp'} #  The above parameters but with equal temperatures and Zeno limit
    params_big_temp = {'Q_S': 4.33, 'P': 1.04, 'Q_M': 1.51, 'x': 10.0, 'tau': 0.25,\
                   'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_big_temp'} #  The above parameters but with T_M >> T_S
    # Dictionary of the dictionaries of parameters to test
    param_sets = {'opt': params_opt, 'naive': params_naive, 'zeno': params_zeno,\
                   'opt_eq_temp': params_opt_eq_temp, 'zeno_eq_temp': params_zeno_eq_temp,\
                   'opt_uneq_temp': params_opt_uneq_temp}#, 'big_temp': params_big_temp}

    #fun, x = find_pos_net_work(sam)
    #print(f"Maximum net work extraction possible: {fun:.2f} meV")
    #print(f"Optimal parameters: Q_S = {x[0]:.2f}, P = {x[1]:.2f}, Q_M = {x[2]:.2f}, T_M = {x[3]:.2f}, tau = {x[4]:.2f}")
    #fun, x = find_pos_net_work_fixed_temps(sam)
    #print(f"Maximum net work extraction possible with fixed temperatures: {fun:.2f} meV")
    #print(f"Optimal parameters with fixed temperatures: Q_S = {x[0]:.2f}, P = {x[1]:.2f}, Q_M = {x[2]:.2f}, tau = {x[3]:.2f}")

    #params = param_sets['opt_eq_temp']
    #sam.set_Q_S(params['Q_S'])
    #sam.set_P(params['P'])
    #sam.set_Q_M(params['Q_M'])
    #sam.set_tau(params['tau'])
    #sam.set_x(params['x'])
    #fix_n = params['n_prime'] # The meter level to start measuring from. I.e. we measure states fix_n to total_levels
    #file_ending = params['file_ending']+'_test' # The file ending for the data files
    #x_max = 2 # The maximum value of x to test
    #params_vs_temp(sam, temp_range=np.linspace(0, x_max, 100),\
    #                             fname=f"data/params_vs_temp{file_ending}.csv", fixed=fix_n)

    #param_sets = {'opt': params_opt, 'opt_eq_temp': params_opt_eq_temp}
    for params in param_sets.values():
        sam.set_Q_S(params['Q_S'])
        sam.set_P(params['P'])
        sam.set_Q_M(params['Q_M'])
        sam.set_tau(params['tau'])
        fix_n = params['n_prime'] # The meter level to start measuring from. I.e. we measure states fix_n to total_levels
        file_ending = params['file_ending'] # The file ending for the data files
        x_max = 2.0 # The maximum value of x to test
        params_vs_temp(sam, temp_range=np.linspace(1e-2, x_max, 100),\
                        fname=f"data/params_vs_temp{file_ending}.csv", fixed=fix_n)
        sam.set_x(params['x'])
        params_vs_omega_per_delta_E(sam, omega_range=np.linspace(1e-1, x_max*sam.get_Q_S(), 100),\
                                     fname=f"data/params_vs_omega_per_delta_E{file_ending}.csv", fixed=fix_n)
        sam.set_Q_M(params['Q_M'])
        params_vs_time(sam, tau_range=np.linspace(0, 2.0, 100),\
                        fname=f"data/params_vs_time{file_ending}.csv", fixed=fix_n)
        sam.set_tau(params['tau'])
        #params_vs_coupling(sam, g_range=np.linspace(0.0, x_max, 100),\
        #                    fname=f"data/params_vs_coupling{file_ending}.csv", fixed=fix_n)
        #params_vs_nprime(sam, nprime_range=np.arange(0, sam.get_total_levels()),\
        #                fname=f"data/params_vs_nprime{file_ending}.csv")

def plot_cond_prob(sam, times=[0.5, 0.75, 1.0], fname=None):
    """ Plots the conditional probabilities as a function of time measuring in state n.
        Here n is the measurement state of the meter.
        
        Args:
            sam (SystemAndMeter): The system and meter object.
            times (ndarray or list, optional): The times at which to evaluate the conditional probabilities. Defaults to [0.5, 0.75, 1.0].
            fname (str, optional): The filename to save the plot. Defaults to None.
            """
    fig, axs = plt.subplots(len(times), 1, figsize=(10, 8))
    for i,t in enumerate(times):
        sam.set_time(t)
        p0, p1 = sam.joint_prob_evol()
        # Calculate conditional probabilities without calling the function for efficiency
        p0_cond = p0/(p0+p1)
        p1_cond = p1/(p0+p1)
        axs[i].plot(np.arange(0, len(p0))*t/len(p0), p0_cond, label=rf'Cond prob $P_0(t|n)$', color='blue')
        axs[i].plot(np.arange(0, len(p1))*t/len(p1), p1_cond, label=rf'Cond prob $P_1(t|n)$', color='red')
        axs[i].set_title(f'Conditional Probabilities as a function of time measuring in state {sam.get_n()}, t_max={t:.2f}')
        axs[i].set_xlabel(r'Reduced time, $[2\pi/\omega]$')
        axs[i].set_ylabel('Probability')
        axs[i].legend()

    plt.tight_layout()
    if fname is None:
        plt.savefig(f'../images/conditional_probabilities_Tm_{sam.get_temp_meter()}.png')
    else:
        plt.savefig(fname)
    print("Conditional probabilities plotted")

def plot_joint_probabilities(sam, time = 0.9, fname=None):
    """Plots the joint probabilities as a function of meter level at a given time.

    Args:
        sam (SystemAndMeter): The system and meter object.
        time (float, optional): The time at which to evaluate hte joint probabilities. Defaults to 0.9.
        fname (str, optional): The filename to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots()
    if isinstance(time, (np.ndarray, list)):
        time = time[0]
    sam.set_time(time)

    meas_levels = np.arange(0, sam.get_total_levels()+1)
    joint_probs = np.zeros_like(meas_levels, dtype=np.float64)
    joint_probs_0 = np.zeros_like(meas_levels, dtype=np.float64)
    cond_probs = np.zeros_like(meas_levels, dtype=np.float64)
    for i in range(len(meas_levels)):
        sam.set_n(i) 
        p0_n, p1_n = sam.joint_probability(sam.get_n(), sam.get_time())
        joint_probs[i] = p1_n
        joint_probs_0[i] = p0_n
        cond_probs[i] = p1_n/(p0_n+p1_n)
    ax.scatter(meas_levels, joint_probs, s=10, color='red', label='Joint prob P_1(n,t)')
    ax.scatter(meas_levels, joint_probs_0, s=10, color='blue', label='Joint prob P_0(n,t)')
    ax.scatter(meas_levels, cond_probs, s=10, color='green', label='Cond prob P(1|n,t)')
    ax.set_xlabel('Meter Level')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.set_title(fr'Prob of i given meter in n time $t={time:.3f}')
    if fname is None:
        plt.savefig(f'../images/joint_probabilities_Tm_{sam.get_temp_meter()}_tm_{time}.png')
    else:
        plt.savefig(fname)
    print("Joint probabilities plotted")

def plot_cond_entropy(sam, times=[0.5, 0.75, 1.0], fname=None):
    """Plots the conditional entropy as a function of meter level at different times.

    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the conditional entropy. Defaults to [0.5, 0.75, 1.0].
        fname (str, optional): The filename to save the plot. Defaults to None.
    """
    fig, axs = plt.subplots(len(times),1)
    for i,t in enumerate(times):
        cond_entropy = np.zeros(sam.get_total_levels(), dtype=np.float64)
        for n in range(sam.get_total_levels()):
            cond_entropy[n] = sam.conditional_entropy(n=n, time=t)
        axs[i].scatter(np.arange(0, sam.get_total_levels()), cond_entropy, label=f'$S_n(t={t:.3f})$')
        axs[i].set_xlabel('Meter Level')
        axs[i].set_ylabel('Conditional Entropy')
        axs[i].set_title(f'Conditional Entropy as a function of meter level at time t={t:.3f}')
        axs[i].legend()
    plt.tight_layout()
    if fname is None:
        plt.savefig(f'../images/conditional_entropy_Tm_{sam.get_temp_meter()}.png')
    else:
        plt.savefig(fname)
    print("Conditional entropy plotted")

def plot_entropy(sam, times=np.linspace(0.0, 2, 100), fname=None):
    """Plots the entropy as a function of time.

    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the entropy. Defaults to np.linspace(0.0, 2, 100).
        fname (str, optional): The filename to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots()
    entropy = np.zeros_like(times)
    cond_entropy = np.zeros_like(times)
    sam.set_n(0)
    for i,t in enumerate(times):
        sam.set_time(t)
        entropy[i] = sam.entropy()
    ax.plot(times, entropy, color='blue', label='Entropy')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy as a function of time')
    if fname is None:
        plt.savefig(f'../images/entropy_Tm_{sam.get_temp_meter()}.png')
    else:
        plt.savefig(fname)
    print("Entropy plotted")

def plot_mutual_info(sam, times=np.linspace(0.0, 2, 100), fname=None):
    """Plots the mutual information between the system and meter as a function of time.

    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the mutual information. Defaults to np.linspace(0.0, 2, 100).
        fname (str, optional): The filename to save the plot. Defaults to None.
    """
    # Plot mutual information as a function of time
    mut_info = np.zeros_like(times)
    fig, ax = plt.subplots()
    for i, t in enumerate(times):
        sam.set_tau(t)
        sam.full_update()
        sam.set_n(first_positive_W_ext(sam))
        sam.set_n_upper_limit(sam.get_total_levels())
        mut_info[i] = sam.mutual_information(time = t)
    ax.plot(times, mut_info, color='blue', label='Mutual Information')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mutual Information')
    ax.set_title('Mutual information between system and meter as a function of time')
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.savefig(f'../images/mutual_information_Tm_{sam.get_temp_meter()}.png')
    print("Mutual information plotted")

def plot_observer_info(sam, times=[0.9], fname=None):
    """Plots the observer information as a function of meter level at different times.
    
    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the observer information. Defaults to [0.9].
        fname (str, optional): The filename to save the plot. Defaults to None.
    """
    fig, axs = plt.subplots(len(times), 1)
    if len(times) == 1:
        axs = [axs]
    for ax, t in zip(axs, times):
        sam.set_time(t)
        for n in range(sam.get_total_levels()):
            sam.set_n(n)
            obs_info = sam.observer_information()
            ax.scatter(n, obs_info, color='blue', label='Observer Information')
        ax.set_xlabel('Meter Level')
        ax.set_ylabel('Observer Information')
        ax.set_title('Observer Information as a function of meter level')
    if fname is None:
        plt.savefig(f'../images/observer_information_Tm_{sam.get_temp_meter()}.png')
    else:
        plt.savefig(fname)
    print("Observer information plotted")

def plot_observer_info_vs_time(sam, times=np.linspace(0.0, 2, 100), fname=None):
    """Plots the observer information as a function of time.

    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the observer information. Defaults to np.linspace(0.0, 2, 100).
        fname (str, optional): The filename to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots()
    obs_info = np.zeros_like(times)
    for i, t in enumerate(times):
        sam.set_tau(t)
        sam.full_update()
        sam.set_n(first_positive_W_ext(sam))
        sam.set_n(first_positive_W_ext(sam))
        sam.set_n_upper_limit(sam.get_total_levels())
        obs_info[i] = sam.observer_information()
    ax.plot(times, obs_info, color='blue', label='Observer Information')
    ax.set_xlabel('Time')
    ax.set_ylabel('Observer Information')
    ax.set_title('Observer Information as a function of time')
    plt.tight_layout()
    if fname is None:
        plt.savefig(f'../images/observer_information_vs_time_Tm_{sam.get_temp_meter()}.png')
    else:
        plt.savefig(fname)
    print("Observer information plotted")

def plot_work(sam, times=np.linspace(0.0, 2, 100), work_type='extracted', sep=False, fname=None):
    """Plots work as a function of time, either extracted, measurement, or both.
        Can plot the extracted and measurement work in separate plots if sep is True.

    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray, optional): The times at which to evaluate the work. Defaults to np.linspace(0.0, 2, 100).
        work_type (str, optional): The type of plot you want, valid options are 'extracted', 'measurement', 'both'. Defaults to 'extracted'.
        sep (bool, optional): Whether to plot in separate plots. Defaults to False.
        fname (str, optional): The filename to save the plot. Defaults to None.
    """
    if fname is None:
        fname = f'../images/{work_type}_work_Tm_{sam.get_temp_meter()}'
    # If sep is True but only one type of work is requested, set sep to False
    if sep and work_type != 'both':
        sep = False
    # If sep is True, create two subplots, otherwise create one
    if sep:
        fig, axs = plt.subplots(2, 1)
        types = ['Extracted', 'Measurement']
    else:
        fig, ax = plt.subplots()
    # Calculate the work at each time
    work = np.zeros_like(times)
    # Calculate the measurement work at each time
    work_meas = np.zeros_like(times)
    for i, t in enumerate(times):
        sam.set_tau(t)
        work[i] = sam.work_extraction()
        work_meas[i] = sam.work_measurement()
    if sep:
        for ax, w, typ in zip(axs, [work, work_meas], types):
            ax.plot(times, w, color='blue', label=f'{typ.capitalize()} Work')
            ax.set_xlabel('Time')
            ax.set_ylabel('Work [meV]')
            ax.set_title(f'{typ.capitalize()} Work as a function of time')
    else:
        if work_type == 'extracted':
            ax.plot(times, work, color='blue', label='Extracted Work')
        elif work_type == 'measurement':
            ax.plot(times, work_meas, color='red', label='Measurement Work')
        elif work_type == 'both':
            ax.plot(times, work, color='blue', label=r'$W_{ext}$')
            ax.plot(times, work_meas, color='red', label=r'$W_{meas}$')
        ax.set_title(f'{work_type.capitalize()} Work as a function of time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Work [meV]')
        ax.legend()
    plt.tight_layout()
    if sep:
        plt.savefig(f'{fname}_sep.png')
    else:
        plt.savefig(f'{fname}.png')
    print(f"{work_type.capitalize()} work plotted")

def plot_quality(sam, times=np.linspace(0.0, 2, 100), fname=None):
    """Plots the quality of the measurement as a function of time.

    Args:
        sam (SystemAndMeter): The coupled system and meter object.
        times (ndarray or list, optional): The times at which to evaluate info and work. Defaults to np.linspace(0.0, 2, 100).
        fname (str, optional): The filename to save the plot. Defaults to None.
    """
    # Plot the quality of the measurement as a function of time
    fig, ax = plt.subplots(3,1)
    quality_work = np.zeros_like(times)
    quality_info_ext = np.zeros_like(times)
    quality_info_msmt = np.zeros_like(times)
    for i, t in enumerate(times):
        sam.set_time(t)
        quality_work[i] = sam.quality_factor()
        quality_info_ext[i], quality_info_msmt[i] = sam.quality_factor_info()
        
    ax[0].plot(times, quality_work, color='blue', label='$W_{meas}/W_{ext}$')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Quality Factor')
    ax[0].set_title('Quality of the measurement as a function of time')
    ax[0].legend()
    ax[1].plot(times, quality_info_ext, color='red', label='$I/W_{ext}$')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Quality Factor')
    ax[1].set_title('Quality of the measurement as a function of time')
    ax[1].legend()
    ax[2].plot(times, quality_info_msmt, color='green', label='$I/W_{meas}$')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Quality Factor')
    ax[2].set_title('Quality of the measurement as a function of time')
    ax[2].legend()
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.savefig(f'../images/quality_Tm_{sam.get_temp_meter()}.png')
    print("Quality plotted")

def plot_quality_coupling(sam, coupling_strengths=[0.0, 1.0, 10.0]):
    """Plots the quality of the measurement as a function of coupling strength.

    Args:
        sam (SystemAndMeter): The coupled system and meter object.
        coupling_strengths (ndarray or list, optional): The coupling strengths at which to evaluate the quality. Defaults to [0.0, 1.0, 10.0].
    """
    for P in coupling_strengths:
        sam.set_P(P)
        plot_quality(sam, fname=f'../images/quality_Tm_{sam.get_temp_meter()}_P_{P}.png')
    sam.set_P(1.0)
    print("Quality plotted for different coupling strengths and P reset to 1.0")

def plot_work_temp(sam, temps=np.linspace(0.0,2.1,100), time=0.5, fname=None):
    """Plots the useful work, W_ext - W_meas, as a function of the normalized temperature x at
        a given normalized time.

    Args:
        sam (SystemAndMeter): The coupled system and meter object.
        temps (ndarray or list, optional): The temperatures to evaluate at. Defaults to np.linspace(0,2,100).
        time (float, optional): The normalized time at which to evaluate the work. Defaults to 0.5.
        fname (str, optional): The file name for the plot. Defaults to None.
    """
    old_time = sam.get_tau()
    old_x = sam.get_x()
    sam.set_tau(time)
    W_ext = np.zeros_like(temps)
    W_meas = np.zeros_like(temps)
    for i,x in enumerate(temps):
        sam.set_x(x)
        W_ext[i] = sam.work_extraction()
        W_meas[i] = sam.work_measurement()
    fig, ax = plt.subplots()
    ax.plot(temps, W_ext, color='blue', label=r'$W_{ext}$')
    ax.plot(temps, W_meas, color='red', label=r'$W_{meas}$')
    ax.plot(temps, W_ext-W_meas, color='green', label='$W_{ext} - W_{meas}$')
    ax.set_xlabel(r'Normalized Temperature, $T_{meter}/T_{system}$')
    ax.set_ylabel('Work [meV]')
    ax.set_title(f'Work at {sam.get_tau()} period of the meter, system at T={sam.get_temp_system()}K')
    ax.legend()
    plt.tight_layout()
    if fname is None:
        plt.savefig(f'../images/work_temp_Tm_{sam.get_temp_meter()}_t_{time}.png')
    else:
        plt.savefig(fname)
    sam.set_tau(old_time)
    sam.set_x(old_x)
    print(f"W_ext and W_meas plotted at time {time}, and x reset to {old_x} and time reset to {old_time}")

def plot_Wmeas_vs_I_mutual(sam, times=np.linspace(0.0, 2, 100), fname=None):
    """Plots the measurement work versus the mutual information as a function of time.

    Args:
        sam (SystemAndMeter): The coupled system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the work and mutual information. Defaults to np.linspace(0.0, 2, 100).
        fname (str, optional): The filename to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots()

    W_meas = np.zeros_like(times)
    I = np.zeros_like(times)
    for i, t in enumerate(times):
        sam.set_time(t)
        W_meas[i] = sam.work_measurement()
        I[i] = sam.mutual_information()

    # Rescale times to show the extracted work
    ax.plot(I[len(I)//2:], W_meas[len(W_meas)//2:], color='blue', label=r'$I$')
    ax.set_xlabel(r'Mutual Information, $I(\tau)$')
    ax.set_ylabel(r'$W_{meas}(\tau)$ [meV]')
    ax.set_title('Measurement work as a function of mutual information')

    # Set the y-ticks to the times points and the y-tick labels to the corresponding measurement work
    #y_ticks = ax.get_yticks()
    #y_tick_labels = [f'{W_meas[np.abs(times - yt).argmin()]:.2f}' for yt in y_ticks]
    #ax.set_yticklabels(y_tick_labels)  
    
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)
    else:
        plt.savefig(f'../images/Wmeas_vs_I_Tm_{sam.get_temp_meter()}.png')
    print("W_meas plotted against I")

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
            sam (SystemAndMeter): The coupled system and meter object.
        
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
def find_pos_net_work(sam):
    """ Find the maximum net work extraction possible for the given system and meter.

    Args:
        sam (SystemAndMeter): The coupled system and meter object.

    Returns:
        float: The maximum net work extraction possible."""
    from scipy.optimize import minimize
    # Set the initial guess for the minimizer and bounds that are non-zero and positive
    x0 = [sam.get_Q_S(), sam.get_P(), sam.get_Q_M(), sam.get_x(), sam.get_tau()]
    sam.set_n(first_positive_W_ext(sam))
    # Add some small random noise to the initial guess to avoid getting stuck in local minima
    x0 = [x + np.random.normal(-0.1, 0.1) for x in x0]
    res = minimize(work_minimizer, x0, args=(sam), bounds=[(1e-2, None), (1e-2, None), (0.2, None), (1e-2, None), (0,1)], method='L-BFGS-B')
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
    W_ext = sam.work_extraction()
    W_meas = sam.work_measurement()
    return -(W_ext - W_meas)
def find_pos_net_work_fixed_temps(sam: SystemAndMeter):
    from scipy.optimize import minimize
    # Set the initial guess for the minimizer and bounds that are non-zero and positive
    sam.set_x(1.0)
    sam.set_n(1)
    x0 = [sam.get_Q_S(), sam.get_P(), sam.get_Q_M(), sam.get_tau()]
    # Add some small random noise to the initial guess to avoid getting stuck in local minima
    x0 = [x + np.random.normal(-0.1, 0.1) for x in x0]
    res = minimize(work_minimizer_fixed_temps, x0, args=(sam), bounds=[(1e-2, None), (1e-2, None), (1e-2, None), (0, 1)], method='L-BFGS-B')
    return -res.fun, res.x

def params_vs_temp(sam: SystemAndMeter, temp_range=np.linspace(0.0, 2.0, 100), fname="data/params_vs_temp.csv", fixed=None, n_upper_limit=None):
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
    hbar = 1e3*sp.constants.physical_constants['reduced Planck constant in eV s'][0]# Reduced Planck constant in meV s
    for omega in omega_range:
        sam.set_Q_M(omega)
        sam.full_update()
        if fixed is None:
            n = first_positive_W_ext(sam)
        else:
            n = fixed
        sam.set_n(n)
        sam.set_n_upper_limit(sam.get_total_levels())
        hw_per_delta_E = hbar*sam.get_omega()/sam.get_delta_E()
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

    # Write the header lines manually
    with open(fname, mode="w") as file:
        file.write(f"System temperature: {sam.get_temp_system():.3f},\
                    Coupling strength: {sam.get_P():.3f}, \
                        Period: {sam.get_tau():.3f}, \
                            Temperature: {sam.get_x():.3f}\n")
    # Append the data to the file
    df.to_csv(fname, mode='a', index=False)
    print(f"Parameters vs omega/delta_E saved to {fname}")

def params_vs_time(sam: SystemAndMeter, tau_range=np.linspace(0.0, 2.0, 100), fname="data/params_vs_time.csv", fixed=None):
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


if __name__=='__main__':
    main()

