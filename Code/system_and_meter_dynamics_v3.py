import numpy as np
from SystemAndMeter_v3 import SystemAndMeter
import pandas as pd

######
# This is the script used to generate the data for the article.
# It contains functions to generate data for the meter level as a function of temperature, work against time, heatmaps,
# thermodynamic and information efficiencies against temperature, conditional probabilities against the meter level, and thermodynamic and information efficiencies against time.
# The functions named phase_boundary_multidata etc are used to generate the heatmaps in the article. 
# The names of the functions might seem a bit arcane, but they came from being reworked old functions that served a slightly different purpose.
#
# The script relies on the SystemAndMeter_v3.py file, which contains the SystemAndMeter class.
######

kB = 1
hbar = 1
save_dir = 'data/article_data_v3/testing/' # to save the data 

def main():
    temp_system = 1.0
    x = 1.0
    Q_S = 1.0
    Q_M = 1.0
    P = 1.0
    msmt_state = 0
    sam = SystemAndMeter(T_S=temp_system, x=x, Q_S=Q_S, Q_M=Q_M, P=P, msmt_state=msmt_state)
    # Dictionaries of parameters to test. They are here to ensure consistency in the parameters.
    # Either select the one you want to use or make your own.
    params_article = {'Q_S': 4.00, 'P': 1., 'Q_M': 1.5, 'x': 0.1, 'tau': 0.25,\
                      'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_article'} # Parameters from the article
    params_All_Ones = {'Q_S': 1, 'P': 0.1, 'Q_M': 0.1, 'x': 0.1, 'tau': 0.25,\
                      'n_prime': int(1), 'n_upper_limit': None, 'file_ending': '_article'} # Parameters from the article
    
    
    
    params = params_article
    sam = SystemAndMeter(T_S=temp_system, x=params['x'], Q_S=params['Q_S'], Q_M=params['Q_M'], P=params['P'], msmt_state=params['n_prime'])
    ## ---------------- Data for the meter level as function of temperature ----------------
    ## Reset all parameters to ensure a consistent initial state
    # Sweep meter levels over a temperature range and record where p(1|n) > p(0|n)
    meter_levels = []
    temp_range = np.linspace(0.0, 2.0, 1000)

    for T in temp_range:
        sam.set_x(T)
        sam.full_update()

        # Search for the first meter level n where p(1|n, T) > p(0|n, T)
        for n in range(sam.get_total_levels()):
            p0, p1 = sam.conditional_probability(n=n)
            if p1 > p0:
                meter_levels.append(n)
                break
        else:
            # If no level satisfies p1 > p0, append a sentinel (e.g., -1)
            meter_levels.append(-1)

    # Save results to CSV
    df = pd.DataFrame({
        'Temperature': temp_range,
        'Meter Level': meter_levels
    })
    df.to_csv(f"{save_dir}meter_level_vs_temp_v3.csv", index=False)


    sam.set_x(params['x'])

    ### ----------- Data for the work against time function  -----------
    ## Reset all the parameters again to make sure we start from the same point
    sam.set_Q_S(params['Q_S'])
    sam.set_Q_M(params['Q_M'])
    sam.set_P(params['P'])
    sam.set_x(params['x'])
    sam.set_tau(params['tau'])
    sam.set_n(params['n_prime'])
    sam.set_n_upper_limit(params['n_upper_limit'])
    sam.set_R(0.0)
    tau_vals = np.linspace(0.0, 5.0, 1000)
    ratios = [0.01, 0.1, 1.0]
    for ratio in ratios:
        # g_eff^2 = P^2 *k_B * T_S and Delta E = k_B * T_S * Q_S
        # Thus we have g_eff^2/Delta E = P^2/Q_S = ratio
        # Therefore we can set P = np.sqrt(ratio * Q_S)
        sam.set_P(np.sqrt(ratio*sam.get_Q_S())) 
        sam.full_update()
        params_vs_time(sam, tau_range=tau_vals, fname=f'{save_dir}params_vs_time_ratio={ratio}.csv')
    # Also make a run with x=0.2
    sam.set_x(0.2)
    sam.full_update()
    params_vs_time(sam, tau_range=tau_vals, fname=f'{save_dir}params_vs_time_ratio={ratio}_x=0.2.csv')
    # #--------------------------------------------------------------------------------

    ## ----------- Data for the heatmaps -----------
    # Fair warning: This takes quite a while to run. About 2-3 hours on the machine I used to run it.
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
    lim_vals = [(1e-6, 50), (1e-6, 5), (1e-6,5), (1e-6,5)]
    for tau, lims in zip(tau_vals, lim_vals):
        sam.set_tau(tau)
        sam.full_update()
        # P^2/Q_S = y => P = sqrt(y * Q_S) where y is the ratio of the effective coupling strength to the system energy splitting
        P_range = np.linspace(np.sqrt(lim_vals[0]*sam.get_Q_S()), np.sqrt(lims[1]*sam.get_Q_S()), 500)  
        # Heatmap for the ergotropy
        phase_boundary_multidata_coupling(sam, temp_range=np.linspace(0.0, 2.0, 1000),\
                                           fname=f'phase_boundary_multidata_tau={tau}_R=0.0.csv',\
                                              work_type='ergotropy', P_range=P_range)
    ## --------------------------------------------------------------------------------

    ## -------- Quantities as a function of relative temperature -----------
    ## Reset all the parameters again to make sure we start from the same point
    sam.set_Q_S(params['Q_S'])
    sam.set_Q_M(params['Q_M'])
    sam.set_P(np.sqrt(0.1*sam.get_Q_S()))  # Set P such that P^2/Q_S = 0.1
    sam.set_x(params['x'])
    sam.set_tau(params['tau'])
    sam.set_n(params['n_prime'])
    sam.set_n_upper_limit(params['n_upper_limit'])
    sam.set_R(0.0)
    sam.full_update()

    temp_range = np.linspace(0.0, 1.0, 1000)
    params_vs_temp(sam, temp_range=temp_range, fname=f'{save_dir}params_vs_temp_v3.csv', type='ergotropy')
    ## ----------------------------------------------

    ## --------------------- Data for the conditional probabilities against the meter level ------------------
    # Reset all the parameters again to make sure we start from the same point
    sam.set_Q_S(1.0)
    sam.set_Q_M(sam.get_Q_S()*0.1)
    sam.set_P(1.0)
    sam.set_x(0.3)
    sam.set_tau(0.25)
    sam.set_n(params['n_prime'])
    sam.set_n_upper_limit(params['n_upper_limit'])
    sam.set_R(0.0)
    sam.full_update()

    probabilities_against_meter_level(sam, fname=f'{save_dir}conditional_probabilities_meter_level_v3.csv')
    ## -----------------------------------------------------------------------------------------------------------

    ## --------------- Quantities as a fucntion of time ----------------
    ## Reset all the parameters again to make sure we start from the same point
    params = params_article
    sam.set_Q_S(params['Q_S'])
    sam.set_Q_M(params['Q_M'])
    sam.set_P(np.sqrt(0.1*sam.get_Q_S()))  # Set P such that P^2/Q_S = 0.1
    sam.set_x(params['x'])
    sam.set_tau(params['tau'])
    sam.set_n(params['n_prime'])
    sam.set_n_upper_limit(params['n_upper_limit'])
    sam.set_R(0.0)
    sam.full_update()

    ratios = [0.01, 0.1, 1.0]
    tau_range = np.concatenate( (np.linspace(0.0,0.02,500),np.linspace(0.02, 0.98, 1000),\
                                 np.linspace(0.98,1.02,1000), np.linspace(1.02,1.98,1000),\
                                 np.linspace(1.98,2.0,500)) )
    rel_temps = [0.1, 0.2]
    for x in rel_temps:
        sam.set_x(x)
        sam.full_update()
        for ratio in ratios:
            sam.set_P(np.sqrt(ratio*sam.get_Q_S()))
            sam.full_update()
            params_vs_time(sam, tau_range=tau_range, fname=f'{save_dir}params_vs_time_ratio={ratio}_x={x}.csv', fixed=params['n_prime'], type='ergotropy')
    # -----------------------------------------------------------------------------------------------------------


def params_vs_temp(
    sam: SystemAndMeter,
    temp_range=np.linspace(0.0, 2.0, 100),
    fname="data/params_vs_temp.csv",
    fixed=None,
    n_upper_limit=None,
    type='ergotropy'
):
    """
    Investigates how system observables vary with the meter temperature.

    Parameters:
        sam (SystemAndMeter): Instance of the coupled system and meter.
        temp_range (array-like): Range of meter temperatures to scan.
        fname (str): Path to CSV file to save results.
        fixed (int, optional): Fixed meter level to use. If None, use first level with positive work.
        n_upper_limit (int, optional): Maximum level to consider in work extraction.
        type (str): Either 'ergotropy' or 'excess work'.

    Saves:
        A CSV with columns for work, heat exchanges, and mutual/observer information vs. meter temperature.
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

        # Set meter state
        #if fixed is None:
        #    n = first_positive_W_ext(sam)
        #    sam.set_n(n)
        #else:
        #    sam.set_n(fixed)

        # Set upper limit for meter level exploration
        if n_upper_limit is not None:
            sam.set_n_upper_limit(n_upper_limit)
        else:
            sam.set_n_upper_limit(sam.get_total_levels())

        # Compute thermodynamic quantities
        try:
            if type == 'ergotropy':
                W_ext = sam.ergotropy()
            elif type == 'excess work':
                W_ext = sam.work_extraction_excess()
            else:
                raise ValueError(f"Invalid work type: {type}")
        except Exception as e:
            print(f"[WARNING] Skipping T = {T:.3f}: error computing {type}: {e}")
            continue

        try:
            W_meas = sam.work_measurement()
        except Exception as e:
            print(f"[WARNING] Skipping T = {T:.3f}: error computing measurement work: {e}")
            continue

        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas

        try:
            I_obs = sam.observer_information()
            I_m = sam.mutual_information()
            I = I_obs + I_m
        except Exception as e:
            print(f"[WARNING] Skipping T = {T:.3f}: error computing information: {e}")
            continue

        # Append results
        results['Temperature'].append(T)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)

    # Save to CSV
    df = pd.DataFrame(results)

    with open(fname, mode="w") as file:
        file.write(
            f"# System temperature: {sam.get_temp_system():.3f}, "
            f"Coupling strength: {sam.get_P():.3f}, "
            f"Delta_E: {sam.get_Q_S():.3f}, "
            f"Omega: {sam.get_Q_M():.3f}, "
            f"Period: {sam.get_tau():.3f}\n"
        )
    df.to_csv(fname, mode='a', index=False)
    print(f"[INFO] Parameters vs. temperature saved to {fname}")

def params_vs_time(
    sam: SystemAndMeter,
    tau_range=np.linspace(0.0, 2.0, 100),
    fname="data/params_vs_time.csv",
    fixed=None,
    n_upper_limit=None,
    type='ergotropy'
):
    """
    Investigates how system observables vary with the meter temperature.

    Parameters:
        sam (SystemAndMeter): Instance of the coupled system and meter.
        temp_range (array-like): Range of meter temperatures to scan.
        fname (str): Path to CSV file to save results.
        fixed (int, optional): Fixed meter level to use. If None, use first level with positive work.
        n_upper_limit (int, optional): Maximum level to consider in work extraction.
        type (str): Either 'ergotropy' or 'excess work'.

    Saves:
        A CSV with columns for work, heat exchanges, and mutual/observer information vs. meter temperature.
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

        if tau < 0.3:
            T_M = sam.get_temp_meter()
            omega = sam.get_omega()
            sam.set_n_upper_limit(int(100*np.ceil(kB*T_M/(hbar*omega))+1))
        else:
            sam.update_total_levels()

        # Compute thermodynamic quantities
        try:
            W_ext = sam.work_extraction(work_type=type)
        except Exception as e:
            print(f"[WARNING] Skipping tau = {tau:.3f}: error computing {type}: {e}")
            continue
        #try:
        #    if type == 'ergotropy':
        #        W_ext = sam.ergotropy()
        #    elif type == 'excess work':
        #        W_ext = sam.work_extraction_excess()
        #    else:
        #        raise ValueError(f"Invalid work type: {type}")
        #except Exception as e:
        #    print(f"[WARNING] Skipping tau = {tau:.3f}: error computing {type}: {e}")
        #    continue

        try:
            W_meas = sam.work_measurement()
        except Exception as e:
            print(f"[WARNING] Skipping tau = {tau:.3f}: error computing measurement work: {e}")
            continue

        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas

        try:
            I_obs = sam.observer_information()
            I_m = sam.mutual_information()
            I = I_obs + I_m
        except Exception as e:
            print(f"[WARNING] Skipping tau = {tau:.3f}: error computing information: {e}")
            continue

        # Append results
        results['Time'].append(tau)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)

    # Save to CSV
    df = pd.DataFrame(results)

    with open(fname, mode="w") as file:
        file.write(
            f"# System temperature: {sam.get_temp_system():.3f}, "
            f"Coupling strength: {sam.get_P():.3f}, "
            f"Delta_E: {sam.get_Q_S():.3f}, "
            f"Omega: {sam.get_Q_M():.3f}, "
            f"Temperature: {sam.get_x():.3f}\n"
        )
    df.to_csv(fname, mode='a', index=False)
    print(f"[INFO] Parameters vs. time saved to {fname}")

def params_vs_coupling(
    sam: SystemAndMeter,
    coupling_range=np.linspace(0.1, 10.0, 100),
    fname="data/params_vs_coupling.csv",
    fixed=None,
    n_upper_limit=None,
    type='ergotropy'
):
    """
    Investigates how system observables vary with the coupling strength. To be clear, this varies P where
    g_eff = g*sqrt(m) = P * sqrt(k_B * T_S) where m is the meter mass. 

    So if you want the effective coupling strength g_eff^2 (which is in the article, and has the units of energy),
    you have to account for this in post-processing. Keep this in mind when setting the coupling range.

    Parameters:
        sam (SystemAndMeter): Instance of the coupled system and meter.
        coupling_range (array-like): Range of coupling strengths to scan.
        fname (str): Path to CSV file to save results.
        fixed (int, optional): Fixed meter level to use. If None, use first level with positive work.
        n_upper_limit (int, optional): Maximum level to consider in work extraction.
        type (str): Either 'ergotropy' or 'excess work'.

    Saves:
        A CSV with columns for work, heat exchanges, and mutual/observer information vs. coupling strength.
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

    for P in coupling_range:
        sam.set_P(P)
        sam.full_update()

        # Compute thermodynamic quantities
        try:
            if type == 'ergotropy':
                W_ext = sam.ergotropy()
            elif type == 'excess work':
                W_ext = sam.work_extraction_excess()
            else:
                raise ValueError(f"Invalid work type: {type}")
        except Exception as e:
            print(f"[WARNING] Skipping P = {P:.3f}: error computing {type}: {e}")
            continue

        try:
            W_meas = sam.work_measurement()
        except Exception as e:
            print(f"[WARNING] Skipping P = {P:.3f}: error computing measurement work: {e}")
            continue

        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas

        try:
            I_obs = sam.observer_information()
            I_m = sam.mutual_information()
            I = I_obs + I_m
        except Exception as e:
            print(f"[WARNING] Skipping P = {P:.3f}: error computing information: {e}")
            continue

        # Append results
        results['Coupling Strength'].append(P)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)

    # Save to CSV
    df = pd.DataFrame(results)
    with open(fname, mode="w") as file:
        file.write(
            f"# System temperature: {sam.get_temp_system():.3f}, "
            f"Meter temperature: {sam.get_x():.3f}, "
            f"Delta_E: {sam.get_Q_S():.3f}, "
            f"Omega: {sam.get_Q_M():.3f}, "
            f"Tau: {sam.get_tau():.3f}\n"
        )
    df.to_csv(fname, mode='a', index=False)
    print(f"[INFO] Parameters vs. coupling saved to {fname}")

def params_vs_omega(
    sam: SystemAndMeter,
    omega_range=np.linspace(0.1, 10.0, 100),
    fname="data/params_vs_omega.csv",
    fixed=None,
    n_upper_limit=None,
    type='ergotropy'
):
    """
    Investigates how system observables vary with the meter frequency (omega).
    To be clear, this varies Q_M, where hbar * omega = Q_M * k_B * T_S.

    Parameters:
        sam (SystemAndMeter): Instance of the coupled system and meter.
        omega_range (array-like): Range of frequencies to scan.
        fname (str): Path to CSV file to save results.
        fixed (int, optional): Fixed meter level to use. If None, use first level with positive work.
        n_upper_limit (int, optional): Maximum level to consider in work extraction.
        type (str): Either 'ergotropy' or 'excess work'.

    Saves:
        A CSV with columns for work, heat exchanges, and mutual/observer information vs. frequency.
    """
    results = {
        'Frequency': [],
        "Work": [],
        "System Heat": [],
        "Meter Heat": [],
        "Observer Information": [],
        "Mutual Information": [],
        "Information": []
    }

    for omega in omega_range:
        sam.set_Q_M(omega)
        sam.full_update()

        # Compute thermodynamic quantities
        try:
            if type == 'ergotropy':
                W_ext = sam.ergotropy()
            elif type == 'excess work':
                W_ext = sam.work_extraction_excess()
            else:
                raise ValueError(f"Invalid work type: {type}")
        except Exception as e:
            print(f"[WARNING] Skipping omega = {omega:.3f}: error computing {type}: {e}")
            continue

        try:
            W_meas = sam.work_measurement()
        except Exception as e:
            print(f"[WARNING] Skipping omega = {omega:.3f}: error computing measurement work: {e}")
            continue

        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas

        try:
            I_obs = sam.observer_information()
            I_m = sam.mutual_information()
            I = I_obs + I_m
        except Exception as e:
            print(f"[WARNING] Skipping omega = {omega:.3f}: error computing information: {e}")
            continue

        # Append results
        results['Frequency'].append(omega)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)
    # Save to CSV
    df = pd.DataFrame(results)
    with open(fname, mode="w") as file:
        file.write(
            f"# System temperature: {sam.get_temp_system():.3f}, "
            f"Meter temperature: {sam.get_x():.3f}, "
            f"Delta_E: {sam.get_Q_S():.3f}, "
            f"Coupling strength: {sam.get_P():.3f}, "
            f"Tau: {sam.get_tau():.3f}\n"
        )
    df.to_csv(fname, mode='a', index=False)
    print(f"[INFO] Parameters vs. omega saved to {fname}")

def params_vs_delta_E(
    sam: SystemAndMeter,
    delta_E_range=np.linspace(0.1, 10.0, 100),
    fname="data/params_vs_delta_E.csv",
    fixed=None,
    n_upper_limit=None,
    type='ergotropy'
):
    """
    Investigates how system observables vary with the energy difference (Delta_E).
    To be clear, this varies Q_S, where Delta_E = k_B * T_S * Q_S.

    Parameters:
        sam (SystemAndMeter): Instance of the coupled system and meter.
        delta_E_range (array-like): Range of energy differences to scan.
        fname (str): Path to CSV file to save results.
        fixed (int, optional): Fixed meter level to use. If None, use first level with positive work.
        n_upper_limit (int, optional): Maximum level to consider in work extraction.
        type (str): Either 'ergotropy' or 'excess work'.

    Saves:
        A CSV with columns for work, heat exchanges, and mutual/observer information vs. energy difference.
    """
    results = {
        'Delta_E': [],
        "Work": [],
        "System Heat": [],
        "Meter Heat": [],
        "Observer Information": [],
        "Mutual Information": [],
        "Information": []
    }

    for delta_E in delta_E_range:
        sam.set_Q_S(delta_E)
        sam.full_update()

        # Compute thermodynamic quantities
        try:
            if type == 'ergotropy':
                W_ext = sam.ergotropy()
            elif type == 'excess work':
                W_ext = sam.work_extraction_excess()
            else:
                raise ValueError(f"Invalid work type: {type}")
        except Exception as e:
            print(f"[WARNING] Skipping Delta_E = {delta_E:.3f}: error computing {type}: {e}")
            continue

        try:
            W_meas = sam.work_measurement()
        except Exception as e:
            print(f"[WARNING] Skipping Delta_E = {delta_E:.3f}: error computing measurement work: {e}")
            continue

        W = W_ext - W_meas
        Q_S = -W_ext
        Q_M = W_meas

        try:
            I_obs = sam.observer_information()
            I_m = sam.mutual_information()
            I = I_obs + I_m
        except Exception as e:
            print(f"[WARNING] Skipping Delta_E = {delta_E:.3f}: error computing information: {e}")
            continue

        # Append results
        results['Delta_E'].append(delta_E)
        results['Work'].append(W)
        results['System Heat'].append(Q_S)
        results['Meter Heat'].append(Q_M)
        results['Observer Information'].append(I_obs)
        results['Mutual Information'].append(I_m)
        results['Information'].append(I)
    # Save to CSV
    df = pd.DataFrame(results)
    with open(fname, mode="w") as file:
        file.write(
            f"# System temperature: {sam.get_temp_system():.3f}, "
            f"Meter temperature: {sam.get_x():.3f}, "
            f"Coupling strength: {sam.get_P():.3f}, "
            f"Omega: {sam.get_Q_M():.3f}, "
            f"Tau: {sam.get_tau():.3f}\n"
        )
    df.to_csv(fname, mode='a', index=False)
    print(f"[INFO] Parameters vs. Delta_E saved to {fname}")

def phase_boundary_multidata_coupling(sam: SystemAndMeter,
                                      temp_range=np.linspace(1e-2, 2.0, 100),
                                      P_range=np.linspace(0.1, 10.0, 500),
                                      fname="data/phase_boundary_coupling.csv",
                                      work_type="ergotropy"):
    """
    Investigate the phase boundary by evaluating net work across varying coupling strengths
    (P² / Q_S) for a range of temperatures.

    Produces a long-form CSV table with columns:
    T, P2_over_QS, W, W_ext, W_meas

    Args:
        sam (SystemAndMeter): Coupled system-meter object.
        temp_range (array-like): Array of temperature values.
        fname (str): Output CSV file name.
        work_type (str): Type of work ('ergotropy' or 'excess work').
    """
    Q_S = sam.get_Q_S()

    # Prepare a list of results to build the DataFrame
    records = []

    for P in P_range:
        sam.set_P(P)
        sam.full_update()

        for T in temp_range:
            sam.set_x(T)
            sam.full_update()

            W_ext = sam.work_extraction(work_type=work_type)
            W_meas = sam.work_measurement()
            W = W_ext - W_meas

            records.append({
                "T": T,
                "P2_over_QS": P**2 / Q_S,
                "W": W,
                "W_ext": W_ext,
                "W_meas": W_meas
            })

    # Convert to DataFrame and write to CSV
    df = pd.DataFrame(records)
    df.to_csv(fname, index=False)
    print(f"Phase boundary multidata saved to {fname}")

def phase_boundary_multidata(sam: SystemAndMeter,
                            temp_range=np.linspace(1e-2, 2.0, 100),
                            Q_M_range=np.linspace(0.1, 10.0, 500),
                            fname="data/phase_boundary.csv",
                            work_type="ergotropy"):
    """    Investigate the phase boundary by evaluating net work across varying energy splitting in the meter
        (hbar * omega / Delta_E = Q_M/Q_S)  for a range of temperatures.

    Produces a long-form CSV table with columns:
        T, Q_M_over_Q_S, W, W_ext, W_meas
    Args:
        sam (SystemAndMeter): Coupled system-meter object.
        temp_range (array-like): Array of temperature values.
        fname (str): Output CSV file name.
        work_type (str): Type of work ('ergotropy' or 'excess work').
    """
    Q_S = sam.get_Q_S()
    # Prepare a list of results to build the DataFrame
    records = []
    for Q_M in Q_M_range:
        sam.set_Q_M(Q_M)
        sam.full_update()

        for T in temp_range:
            sam.set_x(T)
            sam.full_update()

            W_ext = sam.work_extraction(work_type=work_type)
            W_meas = sam.work_measurement()
            W = W_ext - W_meas

            records.append({
                "T": T,
                "Q_M_over_Q_S": Q_M / Q_S,
                "W": W,
                "W_ext": W_ext,
                "W_meas": W_meas
            })
    # Convert to DataFrame and write to CSV
    df = pd.DataFrame(records)
    df.to_csv(fname, index=False)
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


if __name__=='__main__':
    main()

