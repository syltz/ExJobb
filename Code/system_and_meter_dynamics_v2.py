import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from SystemAndMeter_v2 import SystemAndMeter


# Throughout this code, the meter is assumed to be a harmonic oscillator
# and the system is assumed to be a two-level system.
# I also use mass = 1, hbar = 1, and kB = 1 for simplicity. I do however keep them
# in the equations for clarity and so that they can be easily changed if needed.
# It's mostly to avoid any potential numerical issues that may arise from the
# small values of hbar and kB.
hbar = 1.0 # sp.constants.hbar
kB = 1.0 # sp.constants.k

def main():
    temp_system = 300.0
    x = 1.0
    Q_S = 1.0
    Q_M = 1.0
    P = 1.0
    msmt_state = 0
    sam = SystemAndMeter(temp_system=temp_system, x=x, Q_S=Q_S, Q_M=Q_M, P=P, msmt_state=msmt_state)
    # Plot the joint probabilities
    #plot_joint_probabilities(sam, time=0.9)
    # Plot the conditional probabilities
    sam.set_n(0)
    #plot_cond_prob(sam)
    plot_cond_entropy(sam)
    #plot_entropy(sam)
    #plot_mutual_info(sam)
    #plot_observer_info(sam)
    #plot_work(sam)
    #plot_quality(sam)
    

def plot_cond_prob(sam, times=[0.5, 0.75, 1.0]):
    """ Plots the conditional probabilities as a function of time measuring in state n.
        Here n is the measurement state of the meter.
        
        Args:
            sam (SystemAndMeter): The system and meter object.
            times (ndarray or list, optional): The times at which to evaluate the conditional probabilities. Defaults to [0.5, 0.75, 1.0]."""
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
    plt.savefig(f'../images/conditional_probabilities_Tm_{sam.get_temp_meter()}.png')
    print("Conditional probabilities plotted")

def plot_joint_probabilities(sam, time = 0.9):
    """Plots the joint probabilities as a function of meter level at a given time.

    Args:
        sam (SystemAndMeter): The system and meter object.
        time (float, optional): The time at which to evaluate hte joint probabilities. Defaults to 0.9.
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
    plt.savefig(f'../images/joint_probabilities_Tm_{sam.get_temp_meter()}_tm_{time}.png')
    print("Joint probabilities plotted")

def plot_cond_entropy(sam, times=[0.5, 0.75, 1.0]):
    """Plots the conditional entropy as a function of meter level at different times.

    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the conditional entropy. Defaults to [0.5, 0.75, 1.0].
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
    plt.savefig(f'../images/conditional_entropy_Tm_{sam.get_temp_meter()}.png')
    print("Conditional entropy plotted")

def plot_entropy(sam, times=np.linspace(0.0, 2, 100)):
    """Plots the entropy as a function of time.

    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the entropy. Defaults to np.linspace(0.0, 2, 100).
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
    plt.savefig(f'../images/entropy_Tm_{sam.get_temp_meter()}.png')
    print("Entropy plotted")

def plot_mutual_info(sam, times=np.linspace(0.0, 2, 100)):
    """Plots the mutual information between the system and meter as a function of time.

    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the mutual information. Defaults to np.linspace(0.0, 2, 100).
    """
    # Plot mutual information as a function of time
    mut_info = np.zeros_like(times)
    fig, ax = plt.subplots()
    for i, t in enumerate(times):
        sam.set_time(t)
        mut_info[i] = sam.mutual_information(time = t)
    ax.plot(times, mut_info, color='blue', label='Mutual Information')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mutual Information')
    ax.set_title('Mutual information between system and meter as a function of time')
    plt.savefig(f'../images/mutual_information_Tm_{sam.get_temp_meter()}.png')
    print("Mutual information plotted")

def plot_observer_info(sam, times=[0.9]):
    """Plots the observer information as a function of meter level at different times.
    
    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray or list, optional): The times at which to evaluate the observer information. Defaults to [0.9].
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
    plt.savefig(f'../images/observer_information_Tm_{sam.get_temp_meter()}.png')
    print("Observer information plotted")

def plot_work(sam, times=np.linspace(0.0, 2, 100), work_type='extracted', sep=False):
    """Plots work as a function of time, either extracted, measurement, or both.
        Can plot the extracted and measurement work in separate plots if sep is True.

    Args:
        sam (SystemAndMeter): The system and meter object.
        times (ndarray, optional): The times at which to evaluate the work. Defaults to np.linspace(0.0, 2, 100).
        work_type (str, optional): The type of plot you want, valid options are 'extracted', 'measurement', 'both'. Defaults to 'extracted'.
        sep (bool, optional): Whether to plot in separate plots. Defaults to False.
    """
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
        sam.set_time(t)
        work[i] = sam.work_extraction()
        work_meas[i] = sam.work_measurement()
    if sep:
        for ax, w, typ in zip(axs, [work, work_meas], types):
            ax.plot(times, w, color='blue', label=f'{typ.capitalize()} Work')
            ax.set_xlabel('Time')
            ax.set_ylabel('Work')
            ax.set_title(f'{typ.capitalize()} Work as a function of time')
    else:
        if work_type == 'extracted':
            ax.plot(times, work, color='blue', label='Extracted Work')
        elif work_type == 'measurement':
            ax.plot(times, work_meas, color='red', label='Measurement Work')
        elif work_type == 'both':
            ax.plot(times, work, color='blue', label='Extracted Work')
            ax.plot(times, work_meas, color='red', label='Measurement Work')
        ax.set_title(f'{work_type.capitalize()} Work as a function of time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Work')
        ax.legend()
    plt.tight_layout()
    if sep:
        plt.savefig(f'../images/{work_type}_work_Tm_{sam.get_temp_meter()}_sep.png')
    else:
        plt.savefig(f'../images/{work_type}_work_Tm_{sam.get_temp_meter()}.png')
    print(f"{work_type.capitalize()} work plotted")

def plot_quality(sam, times=np.linspace(0.0, 2, 100)):
    """Plots the quality of the measurement as a function of time.

    Args:
        sam (SystemAndMeter): The coupled system and meter object.
        times (ndarray or list, optional): The times at which to evaluate info and work. Defaults to np.linspace(0.0, 2, 100).
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
    plt.savefig(f'../images/quality_Tm_{sam.get_temp_meter()}.png')
    print("Quality plotted")

#def plot_quality_coupling(sam):


if __name__=='__main__':
    main()

