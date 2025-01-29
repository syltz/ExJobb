import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import scipy as sp
import scipy.optimize as opt
import os
import re
import numpy as np
import ast
from scipy.interpolate import UnivariateSpline
import csv
import shutil
import gc

# Set the style of the plots
#sns.set_style(style='white')
#sns.set_context(context='paper', font_scale=2)#, rc={"lines.linewidth": 5.0})
#sns.set_palette(palette='colorblind')
lw = 1.75 # Line width

symbol_dict = {'temperature': r'$T_M/T_S$', 'hw/de': r'$\hbar\omega/\Delta E$',\
                'time': r'$\tau=\frac{\omega t}{2\pi}$', 'coupling strength': r'$g$', "work": r'$W$ [meV]',\
                    "system heat": r'$Q_{\text{sys}}$ [meV]', "meter heat": r'$Q_{\text{meter}}$ [meV]',\
                    "information": r'$I$', "observer information": r'$I_{\text{obs}}$',\
                    "mutual information": r'$I_{\text{mut}}$', "entropy": r'$S$',\
                    "measurement work": r'$W_{\text{meas}}$', 'extracted work': r'$W_{\text{ext}}$',\
                    "meter level": r'$n$', "q-factor": r'$Q = \frac{I_m}{W_{meas}}$', "total entropy": r'$S_{\text{tot}}$'}

x_axes = ['Temperature', 'hw/dE', 'Time', 'Coupling Strength', 'Meter Level']
file_endings = ['temp', 'omega_per_delta_E', 'time', 'coupling', 'nprime']

# Defining colors
dark_navy_blue = "#00008B"
deep_forest_green = '#006400'
dark_brown = '#663300'
dark_grey = '#333333'
dark_purple = '#4B0082'
light_grey = '#D3D3D3'
orange = '#FFA500'
dark_red = '#8B0000'

# Color dictionary to be consistent with the colors. Only needs to be used when plotting multiple lines
color_dict = {'Work': dark_navy_blue, 'System Heat': deep_forest_green, 'Meter Heat': dark_brown,\
              'Information': dark_grey, 'Observer Information': dark_purple,\
                  'Mutual Information': light_grey, 'Measurement Work': dark_red}


run_dict = {'opt': '_opt', 'opt_eq_temp': '_opt_eq_temp', 'zeno_eq_temp': '_zeno_eq_temp', \
            'uneq_temp': '_opt_uneq_temp', 'naive': '_naive', 'zeno':'_zeno'}

def main():
    #main_data_list = ['data/phase_boundary_opt_eq_temp_tau=1e-06.csv', 'data/phase_boundary_opt_eq_temp_tau=0.125.csv', 'data/phase_boundary_opt_eq_temp_tau=0.25.csv', 'data/phase_boundary_opt_eq_temp_tau=0.5.csv']
    #ending_data_list = ['data/phase_boundary_opt_eq_temp_tau=1e-06_ending.csv', 'data/phase_boundary_opt_eq_temp_tau=0.125_ending.csv', 'data/phase_boundary_opt_eq_temp_tau=0.25_ending.csv', 'data/phase_boundary_opt_eq_temp_tau=0.5_ending.csv']
    tau_labels = ['1e-06', '0.125', '0.25', '0.5']
    #taus = np.array([1e-06, 0.125, 0.25, 0.5])
    #dissipation_data = [f'data/phase_boundary_dissipation_tau={tau}.csv' for tau in taus]
    #dissipation_data_ending = [f'data/phase_boundary_dissipation_tau={tau}_ending.csv' for tau in taus]
    #unitary_data = [f'data/phase_boundary_unitary_tau={tau}.csv' for tau in taus]
    #unitary_data_ending = [f'data/phase_boundary_unitary_tau={tau}_ending.csv' for tau in taus]
    #dissipation_data_list = phase_diagram_preprocessing(dissipation_data, dissipation_data_ending)
    #unitary_data_list = phase_diagram_preprocessing(unitary_data, ending_data_names=unitary_data_ending)
    #fnames = ['_gamma=0.01.pdf', '_unitary.pdf']
    #labels = [r'$\tau=10^{-6}$', r'$\tau=0.125$', r'$\tau=0.25$', r'$\tau=0.5$']
    #data_sets = [dissipation_data_list, unitary_data_list]
    #for data_set, fname in zip(data_sets, fnames):
        #phase_diagram(data_set, labels=labels, fname=f"thesis_figures/phase_diagram{fname}")
        #phase_diagram_comparison(data_set[0], fname=f"thesis_figures/phase_diagram_comparison{fname}")
    #df_1 = pd.read_csv('data/multidata_Work_tau=1e-06_gamma=0.01.csv', index_col=0)
    #df_2 =  pd.read_csv('data/multidata_Work_tau=0.125_gamma=0.01.csv', index_col=0)
    #df_3 =  pd.read_csv('data/multidata_Work_tau=0.25_gamma=0.01.csv', index_col=0)
    #df_4 =  pd.read_csv('data/multidata_Work_tau=0.5_gamma=0.01.csv', index_col=0)
    #df_list = [df_1, df_2, df_3, df_4]
    #power_heatmap(indata=df_list, times=taus, labels=tau_labels, overlay=True, fname='thesis_figures/power_plot_dissipation.png')
    #gammas = [0, 0.001, 0.01]
    #T_S = []
    #x = []
    #df_list = [pd.read_csv(f'data/dissipation/params_vs_time_opt_eq_temp_gamma_{gamma}_extreme_long_run.csv', skiprows=1) for gamma in gammas]
    ## Use regex to extract the System temperature and the Temperature from the first row of the csv file
    #pattern = r"System temperature: (\d+\.\d+).*?Temperature: (\d+\.\d+)"
    #for gamma in gammas:
    #    with open(f'data/dissipation/params_vs_time_opt_eq_temp_gamma_{gamma}_extreme_long_run.csv', 'r') as f:
    #        reader = csv.reader(f)
    #        first_row = next(reader)
    #        title_string = ', '.join(first_row)
    #        f.close()
    #    match = re.search(pattern, title_string)
    #    T_S.append(float(match.group(1)))
    #    x.append(float(match.group(2)))
    #for df, gamma, T, x_val in zip(df_list, gammas, T_S, x):
    #    df['System Entropy'] = df['System Heat']/T
    #    df['Meter Entropy'] = df['Meter Heat']/(x_val*T)
    #    df['Classical Entropy'] = df['System Entropy'] + df['Meter Entropy']
    #    df['Total Entropy'] = df['Classical Entropy'] + df['Information']
    #    df['No-obs Entropy']= df['Classical Entropy'] + df['Mutual Information']
    #x_axis = 'Time'
    #y_axes = ['Classical Entropy', 'Total Entropy']
    #y_axes.reverse()
    #labels = [r'$\frac{Q_S}{T_S} + \frac{Q_M}{T_M}$', r'$\frac{Q_S}{T_S} + \frac{Q_M}{T_M} + I$']
    #labels.reverse()
    #iv1 = (0.52, 0.56)
    #iv2 = (-0.005, 0.005)
    #plot_broken_y_axis(df_list[0][df_list[0]<2], xaxis=x_axis, yaxis=y_axes, interval_1=iv1, interval_2=iv2, labels=labels, title='Entropy vs Time', xlabel=r'$\tau$',\
    #                    ylabel='Entropy [a.u.]', fname='thesis_figures/entropy_vs_time.png', legend_pos=(0.5, 0.8))
    #plot_multidata(df_list[0],  xaxis=x_axis, yaxis_list=y_axes, labels=[[r'$S_{c}$', r'$S_{c} + I$', r'$S_{c} + I_{mut}$']],\
    #                title='Entropy vs Time', xlabel=r'$\tau$', ylabel='Entropy [a.u.]', fname='thesis_figures/entropy_vs_time_dissipation.png',\
    #                    xlim=(0, 2))
    # Select only the dataframe corresponding to time less than 2
    #print(title_string)
    #df_list = [df[df['Time'] < 2] for df in df_list]
    #plot_entropy(df_list[0], 'dissipation', '0', 'Time', title_string)

    #plot_data_broken_x_axis(df_list, 'Time', 'Information', (0,2), (98,100), fname='thesis_figures/information_vs_time_broken_x_axis.png', labels=[r'$\gamma=0$', r'$\gamma=0.001$', r'$\gamma=0.01$'],\
    #                         title='Information vs Time', xlabel=r'$\tau$', ylabel=r'$I_{tot}$', legend_pos=(-0.08, 0.7))
    #for df in df_list:
    #    df['Q-factor'] = -df['System Heat'] / df['Information']
    #plot_data_broken_x_axis(df_list, 'Time', 'Q-factor', (0,2), (98, 100), fname='thesis_figures/Q_factor_vs_time.png', labels=[r'$\gamma=0$', r'$\gamma=10^{-3}$', r'$\gamma=10^{-2}$'],\
    #                         title='Q-factor vs Time', xlabel=r'Time $\tau$', ylabel=r'$Q = \frac{W_{ext}}{I_{tot}}$', legend_pos=(0.15, 0.802))
    
    #for n in [1,2,3]:
    #    test1_time = pd.read_csv(f'data/test_{n}_vs_time.csv', skiprows=1)
    #    plot_data([test1_time], 'Time', 'Information', f'test_images/test{1}_info_vs_time.png', title='Information vs Time', xlabel=r'$\tau$', ylabel=r'$I_{tot}$')
    #    plot_multidata([test1_time], xaxis='Time', yaxis_list=['Work', 'Meter Heat', 'System Heat'], labels=[['$W$', '$Q_{M}$', '$Q_{S}$']], title='Energy vs Time', xlabel=r'$\tau$', ylabel='Energy [meV]', fname=f'test_images/test{n}_energy_vs_time.png')
    #    test1_temp = pd.read_csv(f'data/test_{n}_vs_temp.csv', skiprows=1)
    #    plot_data([test1_temp], 'Temperature', 'Information', f'test_images/test{n}_info_vs_temp.png', title='Information vs Temperature', xlabel=r'$T_M/T_S$', ylabel=r'$I_{tot}$')
    #    #test1_temp = test1_temp[test1_temp['Temperature'] < 0.5]
    #    plot_multidata([test1_temp], xaxis='Temperature', yaxis_list=['Work', 'Meter Heat', 'System Heat'], labels=[['$W$', '$Q_{M}$', '$Q_{S}$']], title='Energy vs Temperature', xlabel=r'$T_M/T_S$', ylabel='Energy [meV]', fname=f'test_images/test{n}_energy_vs_temp.png')
    #    test1_coupling = pd.read_csv(f'data/test_{n}_vs_coupling.csv', skiprows=1)
    #    plot_data([test1_coupling], 'Coupling Strength', 'Information', f'test_images/test{n}_info_vs_coupling.png', title='Information vs Coupling Strength', xlabel=r'$g$', ylabel=r'$I_{tot}$')
    #    plot_multidata([test1_coupling], xaxis='Coupling Strength', yaxis_list=['Work', 'Meter Heat', 'System Heat'], labels=[['$W$', '$Q_{M}$', '$Q_{S}$']], title='Energy vs Coupling Strength', xlabel=r'$g$', ylabel='Energy [meV]', fname=f'test_images/test{n}_energy_vs_coupling.png')
    #    test1_n = pd.read_csv(f'data/test_{n}_vs_nprime.csv', skiprows=1)
    #    plot_data([test1_n], 'Meter Level', 'Information', f'test_images/test{n}_info_vs_n.png', title='Information vs Meter Level', xlabel=r'$n$', ylabel=r'$I_{tot}$')
    #    plot_multidata([test1_n], xaxis='Meter Level', yaxis_list=['Work', 'Meter Heat', 'System Heat'], labels=[['$W$', '$Q_{M}$', '$Q_{S}$']], title='Energy vs Meter Level', xlabel=r'$n$', ylabel='Energy [meV]', fname=f'test_images/test{n}_energy_vs_n.png')
    #    del test1_time, test1_temp, test1_coupling, test1_n
    #    gc.collect()
    
    #W, W_ext, W_meas = multidata_preprocessing('data/multidata_eq_temp_zeno.csv')
    #W.to_csv('data/multidata_eq_temp_zeno_W.csv')
    #W_ext.to_csv('data/multidata_eq_temp_zeno_W_ext.csv')
    #W_meas.to_csv('data/multidata_eq_temp_zeno_W_meas.csv')
    #W, W_ext, W_meas = multidata_preprocessing('data/multidata_uneq_temp.csv')
    #W.to_csv('data/multidata_uneq_temp_W.csv')
    #W_ext.to_csv('data/multidata_uneq_temp_W_ext.csv')
    #W_meas.to_csv('data/multidata_uneq_temp_W_meas.csv')
    #df_zeno = pd.read_csv('data/multidata_eq_temp_zeno_W.csv', index_col=0)
    ## Select the column where the values of the column header is less than 0.5
    #df_zeno = df_zeno[df_zeno.columns[df_zeno.columns.astype(float) < 0.5]]
    ## Select the rows with indices less than 0.5
    #df_zeno = df_zeno[df_zeno.index.astype(float) < 2.5]
    #df_uneq = pd.read_csv('data/multidata_uneq_temp_W.csv', index_col=0)
    #df_uneq = df_uneq[df_uneq.columns[df_uneq.columns.astype(float) < 0.5]]
    #df_uneq = df_uneq[df_uneq.index.astype(float) < 2.5]
    #power_heatmap([df_zeno, df_uneq], times=[1e-10, 0.1], labels=['Zeno', 'Unequal'], overlay=True, fname='thesis_figures/power_plot_zeno_uneq.png')
    #del df_zeno, df_uneq
    #gc.collect()
    #df = pd.read_csv('data/test_1_vs_temp.csv', skiprows=1)
    # Create a new column for the efficiency / COP
    # When work is positive and temperature is less than 1, the efficiency is Work/abs(System Heat),
    # When work is positive and temperature is greater than 1, the efficiency is (Work + abs(System Heat)/abs(Meter Heat)
    # When work is negative, the efficiency is (System Heat)/abs(Work)
    #df['Efficiency'] = df.apply(calc_efficiency, axis=1)
    ## Plot the efficiency against the temperature
    #plot_data([df], 'Temperature', 'Efficiency', 'thesis_figures/efficiency_vs_temp.png', title='Efficiency vs Temperature', xlabel=r'$T_M/T_S$', ylabel='Efficiency [a.u.]')
    #efficiency_plot(df)
    #df = pd.read_csv('data/test_2_vs_temp.csv', skiprows=1)
    #plt.plot(df['Temperature'], df['Work'])
    #plt.plot(df['Temperature'], df['System Heat'])
    #plt.plot(df['Temperature'], df['Meter Heat'])
    #plt.ylim(-1,1)
    #plt.show()
    #fnames = ['data/multidata_eq_temp_zeno_tau=2e-09.csv', 'data/multidata_eq_temp_zeno_tau=3e-09.csv', 'data/multidata_eq_temp_zeno_tau=4e-09.csv']
    #taus = [2,3,4]
    #for fname, tau in zip(fnames, taus):
    #    W, We, Wm = multidata_preprocessing(fname)
    #    W.to_csv(f'data/multidata_eq_temp_zeno_tau={tau}e-09_W.csv')
    #    We.to_csv(f'data/multidata_eq_temp_zeno_tau={tau}e-09_W_ext.csv')
    #    Wm.to_csv(f'data/multidata_eq_temp_zeno_tau={tau}e-09_W_meas.csv')
    df1 = pd.read_csv('data/multidata_eq_temp_zeno_W.csv', index_col=0)
    df2 = pd.read_csv('data/multidata_eq_temp_zeno_tau=2e-09_W.csv', index_col=0)
    df3 = pd.read_csv('data/multidata_eq_temp_zeno_tau=3e-09_W.csv', index_col=0)
    df4 = pd.read_csv('data/multidata_eq_temp_zeno_tau=4e-09_W.csv', index_col=0)
    df_list = [df1, df2, df3, df4]
    labels = [r'$\tau=10^{-9}', r'$\tau=2\cdot 10^{-9}$', r'$\tau=3\cdot 10^{-9}$', r'$\tau=4\cdot 10^{-9}$']
    power_heatmap(df_list, times=[1e-09, 2e-09, 3e-09, 4e-09], labels=labels, overlay=True, fname='thesis_figures/power_plot_zeno_tau.png', xlim=0.2, ylim=1.5)




# Custom function to preprocess and convert the strings to lists of floats
def convert_to_list(s):
    # Remove the brackets
    s = s.strip('[]')
    # Split the string by spaces and convert to floats
    return [float(x) for x in s.split()]
def plot_all():
    for sim_run in run_dict:
        individual_plots_all(run_dict[sim_run])

def individual_plots_all(sim_run):
    for x_axis, file_ending in zip(x_axes, file_endings):
        # Check if the directory exists
        if not os.path.exists(f'images/param_vs_{file_ending}'):
            os.makedirs(f'images/param_vs_{file_ending}')
        # Import data from csv file
        # Check if the file exists
        try:
            data = pd.read_csv(f'data/params_vs_{file_ending}{sim_run}.csv', skiprows=1)
        except FileNotFoundError:
            print(f"File {file_ending}{sim_run} not found")
            continue
        # Import the fixed parameters and set the figure title
        fixed_params = pd.read_csv(f'data/params_vs_{file_ending}{sim_run}.csv', nrows=1)
        title_string = ', '.join([str(x).strip() for x in fixed_params.columns])
        
        # Plot the parameters against the x-axis
        for param in data.columns[1:]:
            plt.figure(figsize=(12, 8))
            sns.lineplot(x=x_axis, y=param, data=data, linewidth=lw)
            plt.title(title_string)
            # Set x and y labels with the correct symbols
            plt.xlabel(f'{symbol_dict[x_axis.lower()]}')
            plt.ylabel(f'{symbol_dict[param.lower()]}')
            sns.despine()
            plt.savefig(f'images/param_vs_{file_ending}/{param}{sim_run}.png')
            plt.close()
            print(f"Plotted {param} vs {x_axis}")

        # We would like the work comparison to be plotted only against temperature
        if x_axis.lower() == 'temperature':
            plot_work_comparison(data, file_ending, sim_run, x_axis, title_string)
            plot_entropy(data, file_ending, sim_run, x_axis, title_string)
        # We would like to plot the Q-factor for all columns that contain the word "information"
        plot_q_factor(data, file_ending, sim_run, x_axis, title_string)
        if x_axis.lower() == 'time':
            plot_w_meas_vs_mutual_info(data, file_ending, sim_run, x_axis, title_string)

def plot_work_comparison(data, fname, x_axis, title_string):
        # Work = W_meas + W_ext but also stored in the data
        Work = data['Work']
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.lineplot(x=x_axis, y='Work', data=data, label=r'$W$', color=color_dict['Work'], linewidth=lw)
        sns.lineplot(x=x_axis, y='System Heat', data=data, label=r'$Q_{S}$', color=color_dict['System Heat'], linewidth=lw)
        sns.lineplot(x=x_axis, y='Meter Heat', data=data, label=r'$Q_{M}$', color=color_dict['Meter Heat'], linewidth=lw)
        # Color the region 0 to 1 in the x-axis red
        plt.axvspan(0, 1, color='red', alpha=0.3)
        # Color the region 1 to 2 in the x-axis blue
        plt.axvspan(1, 2, color='blue', alpha=0.3)
        # Limit the x-axis to 0 to 2
        plt.xlim(0, 2)
        # Add a horizontal line at 0
        plt.axhline(0, color='black', linewidth=lw)
        plt.legend()
        plt.title(title_string)
        plt.xlabel(f'{symbol_dict[x_axis.lower()]}')
        plt.ylabel('Energy [meV]')
        sns.despine()
        plt.savefig(f'{fname}')
        plt.close()
def plot_q_factor(data, file_ending, sim_run, x_axis, title_string):
    # Select all columns that contain the word "information"
    info_cols = [col for col in data.columns if 'information' in col.lower()]
    # Plot Work / Information for all columns that contain the word "information"
    if info_cols:
        plt.figure(figsize=(12, 8))
        for i, info_col in enumerate(info_cols):
            data[f'{info_col}/Measurement Work'] = data[info_col]/(-data['Meter Heat'])
            plt.subplot(len(info_cols), 1, i + 1)
            sns.lineplot(x=x_axis, y=f'{info_col}/Measurement Work', data=data, linewidth=lw)
            # Set x and y labels with the correct symbols
            plt.xlabel(f'{symbol_dict[x_axis.lower()]}')
            plt.ylabel(f'{symbol_dict[info_col.lower()]} / {symbol_dict["measurement work"]}')
            plt.title(title_string)
            plt.xlabel(f'{x_axis}')
            sns.despine()
        plt.tight_layout()
        plt.savefig(f'images/param_vs_{file_ending}/Q_{info_col}{sim_run}.png')
        plt.close()
        print(f"Plotted {info_col} vs {x_axis}")

def plot_entropy(data, file_ending, sim_run, x_axis, title_string):
    # First extract the system temperature T_S from the title string
    pattern = r"System temperature: (\d+\.\d+)"
    T_S = float(re.search(pattern, title_string).group(1))
    # The x-axis is T_M/T_S so x_axis* T_S = T_M
    data['Meter temperature'] = data[x_axis]*T_S
    # Create vectors for the system and meter heat divided by the system and meter temperature
    data['System Heat / System Temperature'] = data['System Heat']/T_S
    data['Meter Heat / Meter Temperature'] = data['Meter Heat']/data['Meter temperature']
    # Next plot system heat over system temperature and meter heat over meter temperature
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=x_axis, y='System Heat / System Temperature', data=data,\
                  label=r'$\frac{Q_S}{T_S}$',  color=color_dict['System Heat'])
    sns.lineplot(x=x_axis, y='Meter Heat / Meter Temperature', data=data,\
                  label=r'$\frac{Q_M}{T_M}$',  color=color_dict['Meter Heat'])
    # Also plot the different informationsf
    sns.lineplot(x=x_axis, y='Information', data=data, label=r'$I$',\
                   color=color_dict['Information'])
    sns.lineplot(x=x_axis, y='Observer Information', data=data, label=r'$I_{\text{obs}}$',\
                   color=color_dict['Observer Information'])
    sns.lineplot(x=x_axis, y='Mutual Information', data=data, label=r'$I_{\text{mut}}$',\
                   color=color_dict['Mutual Information'])
    # Plot the sum of system heat/system temperature and the mutual information
    data['Total entropy'] = data['System Heat / System Temperature'] + data['Information']
    data['S+I_m']= data['System Heat / System Temperature'] + data['Mutual Information']
    sns.lineplot(x=x_axis, y='Total entropy', data=data, label=r'$\frac{Q_S}{T_S} + I_{mut} + I_{obs}$',\
                   color="black", linestyle='--')
    sns.lineplot(x=x_axis, y='S+I_m', data=data, label=r'$\frac{Q_S}{T_S} + I_{mut}$',\
                     color="black", linestyle='dotted')
    # Add horizontal line at 0
    plt.axhline(0, color='black', linewidth=lw)
    #plt.ylim(-0.01, 0.02)
    plt.title(title_string)
    plt.xlabel(f'{symbol_dict[x_axis.lower()]}')
    plt.ylabel('Entropy [meV/K]')
    plt.legend()
    sns.despine()
    plt.savefig(f'images/testing.png')
    plt.close()

def plot_w_meas_vs_mutual_info(data, file_ending, sim_run, x_axis, title_string):
    # Plot the measurement work against the mutual information
    plt.figure(figsize=(12, 8))
    sns.lineplot(y=-data['System Heat'], x='Mutual Information', data=data, linewidth=lw)
    plt.title(title_string)
    plt.xlabel(f'{symbol_dict["measurement work"]}')
    plt.ylabel(f'{symbol_dict["mutual information"]}')
    sns.despine()
    plt.savefig(f'images/param_vs_{file_ending}/W_meas_vs_I_{sim_run}.png')
    plt.close()

def poster_plotting():
    # Import data from the _opt, _opt_eq_temp and _zeno_eq_temp files
    data_opt = pd.read_csv('data/params_vs_time_opt_testing_2.csv', skiprows=1)
    data_opt_eq_temp = pd.read_csv('data/params_vs_time_opt_eq_temp.csv', skiprows=1)
    data_zeno_eq_temp = pd.read_csv('data/params_vs_time_zeno_eq_temp.csv', skiprows=1)
    color1 = sns.color_palette("colorblind")[0]
    color2 = sns.color_palette("colorblind")[1]
    color3 = sns.color_palette("colorblind")[2]
    color4 = sns.color_palette("colorblind")[3]
    color5 = sns.color_palette("colorblind")[4]
    color6 = sns.color_palette("colorblind")[5]


    # Plot information and work for the three different cases with the y-axis for information
    # on the left and the y-axis for work on the right
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    sns.lineplot(x='Time', y='Information', data=data_opt, label=r'$I(T_M\ll T_S)$', ax=ax1, color='black')
    sns.lineplot(x='Time', y='Meter Heat', data=data_opt, label=r'$W_{meas}(T_M \ll T_S)$', ax=ax2, color=color1,  linestyle='--')
    sns.lineplot(x='Time', y=-data_opt['System Heat'], data=data_opt, label=r'$W_{ext}(T_M \ll T_S)$', ax=ax2, color=color2,  linestyle='--')
    #sns.lineplot(x='Time', y='Information', data=data_opt_eq_temp, label=f'$I(T_M=T_S)$', ax=ax1, color=color2)
    #sns.lineplot(x='Time', y='Meter Heat', data=data_opt_eq_temp, label=r'$W_{meas}(T_M=T_S)$', ax=ax2, color=color2,  linestyle='--')
    ax1.set_ylabel(f'{symbol_dict["information"]}' + ' (a.u.)')
    ax2.set_ylabel(r'$W$'+' (a.u.)')
    ax1.set_xlabel(f'{symbol_dict["time"]}')
    fig.tight_layout()
    #ax2.legend().set_bbox_to_anchor((0.52, 0.1))
    ax1.set_xlim(0,1.01)
    ax2.set_xlim(0,1.01)
    legend1 = ax1.legend()
    #legend1.set_bbox_to_anchor((0.435, 0.3))
    #ax1.legend(loc='center left')
    #ax2.legend(loc='right')
    ax2.legend(loc='lower center')
    ax1.legend().set_bbox_to_anchor((0.62, 0.7))
    #sns.despine()
    plt.savefig('images/poster_plots/poster_info_work.pdf', dpi=300)
    plt.close()

    df_list = [pd.read_csv('data/params_vs_time_opt_eq_temp_gamma_0.csv', skiprows=1),\
                pd.read_csv('data/params_vs_time_opt_eq_temp_gamma_0.1.csv', skiprows=1),\
                    pd.read_csv('data/params_vs_time_opt_eq_temp_gamma_0.01.csv', skiprows=1),\
                        pd.read_csv('data/params_vs_time_opt_eq_temp_gamma_0.001.csv', skiprows=1)]
    labels = [r'$\gamma=0$', r'$\gamma=0.1$', r'$\gamma=0.01$', r'$\gamma=0.001$']
    W_net_per_I_tot(df_list, normalize=True, labels=labels)
    #fig,ax = plt.subplots(figsize=(12, 8))
    #info_per_Wmsmt_opt =(data_opt['Work'])/(data_opt['Information'])
    #info_per_Wmsmt_opt /= info_per_Wmsmt_opt.max()
    #info_per_Wmsm_opt_eq_temp = (data_opt_eq_temp['Work']/(data_opt_eq_temp['Information']))#[1:-1]
    #info_per_Wmsm_opt_eq_temp /= info_per_Wmsm_opt_eq_temp.max()
    ##sns.lineplot(x='Time', y=info_per_Wmsmt_opt, data=data_opt)#, label=r'$\frac{I}{W_{meas}}(T_M\ll T_S)$', color=color1)
    #sns.lineplot(x='Time', y=info_per_Wmsm_opt_eq_temp, data=data_opt_eq_temp, label=r'$\frac{I}{W_{meas}}(T_M=T_S)$', color=color2)
    #plt.title('Net work extracted per unit of information')
    #plt.xlabel(f'{symbol_dict["time"]}')
    #plt.ylabel(r'$W_{net} / I$')
    ##plt.ylabel(f'{symbol_dict["information"]} / '+r'$W_{net}$')
    ##plt.legend()
    ##plt.xlim(-0.02,1.01)
    ##sns.despine()
    #plt.tight_layout()
    #plt.savefig('images/poster_plots/W_net_per_I_tot.pdf', dpi=300)
    #plt.close()
    #fig, ax = plt.subplots(figsize=(12, 8))
    #infor_per_Wmsmt_over_time = (data_opt['Information']/(data_opt['Meter Heat']))[1:-1]
    #infor_per_Wmsmt_over_time /= infor_per_Wmsmt_over_time.max()
    #infor_per_Wmsmt_over_time_eq_temp = (data_opt_eq_temp['Information']/(data_opt_eq_temp['Meter Heat']))[1:-1]
    #infor_per_Wmsmt_over_time_eq_temp /= infor_per_Wmsmt_over_time_eq_temp.max()
    #sns.lineplot(x='Time', y=infor_per_Wmsmt_over_time, data=data_opt, label=r'$\frac{I}{W_{meas}}(T_M\ll T_S)$', color=color1)
    #plt.savefig('images/poster_plots/poster_info_per_Wmsmt_over_time.pdf', format='pdf', dpi=300)

    # Next plot Q-factor I_mutual/W_measurement for all but the zeno case
    data_opt = pd.read_csv('data/params_vs_omega_per_delta_E_opt.csv', skiprows=1)
    data_opt_eq_temp = pd.read_csv('data/params_vs_omega_per_delta_E_opt_eq_temp.csv', skiprows=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    # Add to the dataframe the Q-factor I_mutual/W_measurement
    data_opt['Q-factor'] = data_opt['Mutual Information'] / (data_opt['Meter Heat'])
    Q_fact = data_opt['Q-factor'] / data_opt['Q-factor'].max()
    data_opt_eq_temp['Q-factor'] = data_opt_eq_temp['Mutual Information'] / (data_opt_eq_temp['Meter Heat'])
    Q_fact_eq_temp = data_opt_eq_temp['Q-factor'] / data_opt_eq_temp['Q-factor'].max()
    sns.lineplot(x='hw/dE', y=Q_fact, data=data_opt, label=r'$Q(T_M\ll T_S)$', color=color1)
    sns.lineplot(x='hw/dE', y=Q_fact_eq_temp, data=data_opt_eq_temp, label=r'$Q(T_M=T_S)$', color=color2)
    plt.title('Normalized Q-factor')
    plt.xlabel(f'{symbol_dict["hw/de"]}')
    plt.ylabel(r'$Q / Q_{max}$')
    plt.ylim(0, plt.ylim()[1])
    plt.legend()
    plt.tight_layout()
    sns.despine()
    plt.savefig('images/poster_plots/poster_q_factor.pdf', format='pdf', dpi=300)
    plt.close()

    # Next plot work and heat comparisons for three different cases, opt, opt_eq_temp and zeno_eq_temp
    data_opt = pd.read_csv('data/params_vs_temp_opt_eq_temp.csv', skiprows=1)
    data_zeno = pd.read_csv('data/params_vs_temp_zeno_eq_temp.csv', skiprows=1)
    zeno_corr_factor = 1#e17
    fig, axs = plt.subplots(2,1, figsize=(12, 8), gridspec_kw={'hspace': 0.1})

    

    sns.lineplot(x='Temperature', y='Work', data=data_opt, color=color1, label=r"$W$",  ax=axs[0])
    sns.lineplot(x='Temperature', y=zeno_corr_factor*data_zeno['Work'], data=data_zeno, color=color1, label=r"$W^{Zeno}$",  ax=axs[1])
    sns.lineplot(x='Temperature', y='System Heat', data=data_opt, color=color2, label=r"$Q_S$",  ax=axs[0])
    sns.lineplot(x='Temperature', y=zeno_corr_factor*data_zeno['System Heat'], data=data_zeno, label=r"$Q^{Zeno}_S$", color=color2,  ax=axs[1])
    sns.lineplot(x='Temperature', y='Meter Heat', data=data_opt, label=r"$Q_M$", color=color3,  ax=axs[0])
    sns.lineplot(x='Temperature', y=zeno_corr_factor*data_zeno['Meter Heat'], data=data_zeno, label=r"$Q^{Zeno}_M$", color=color3,  ax=axs[1])

    # Create efficiencies for the two cases
    data_opt['Efficiency'] = data_opt.apply(calc_efficiency, axis=1)
    data_zeno['Efficiency'] = data_zeno.apply(calc_efficiency, axis=1)
    # Add a second axis for the efficiency in both subplots
    ax0_eff = axs[0].twinx()
    #ax1_eff = axs[1].twinx()
    # Color code the efficiency lines so that for TM < TS and W > 0 it is color4, for TM > TS and W > 0 it is color5, else color6
    plot_colored_segments(ax0_eff, data_opt['Temperature'], data_opt['Efficiency'], data_opt['Temperature'] <1, data_opt['Work'] > 0, color4, color5, color6)
    #plot_colored_segments(ax1_eff, data_zeno['Temperature'], data_zeno['Efficiency'], data_zeno['Temperature'] < 1, data_zeno['Work']> 0, color4, color5, color6)
    
    plt.tight_layout()
    # Set y-axis for the efficiency axes
    ax0_eff.set_ylabel(r'$\eta,\, COP$')
    #ax1_eff.set_ylabel(r'$\eta^{Zeno},\, COP$')
    ax0_eff.set_ylim(0, 0.1*data_opt['Efficiency'].max())
    #ax1_eff.set_ylim(0, 0.1*data_zeno['Efficiency'].max())
    # Set y-axis for the work and heat axes
    axs[0].set_ylabel('meV')
    axs[1].set_ylabel(r'meV')

    # Find first x-value when work is negative
    first_neg_work = data_opt[data_opt['Work'] < 0]['Temperature'].values[0]
    axs[0].axvspan(0, 1, color='red', alpha=0.3)
    axs[0].axvspan(1, first_neg_work, color='white', alpha=0.3)
    axs[0].axvspan(first_neg_work, 2, color='blue', alpha=0.3)
    axs[0].axhline(0, color='black', linewidth=lw)

    axs[1].axvspan(0, 1, color='red', alpha=0.3)

    #first_pos_work = data_zeno[data_zeno['Work'] > 0]['Temperature'].values[0]
    #axs[1].axvspan(0, 1, color='yellow', alpha=0.3)
    #axs[1].axvspan(1, first_pos_work, color='blue', alpha=0.3)
    #axs[1].axvspan(first_pos_work, 2, color='white', alpha=0.3)
    #axs[1].axhline(0, color='black', linewidth=lw)


    # Make them share the same x-axis
    plt.setp(axs[0].get_xticklabels(), visible=False)

    # For both axes set y-tick
    ax_0_y_ticks = np.round(np.array([min(data_opt['Work'].min(), data_opt['System Heat'].min(), data_opt['Meter Heat'].min()),\
                    max(data_opt['Work'].max(), data_opt['System Heat'].max(), data_opt['Meter Heat'].max()), 0]),1)
    ax_1_y_ticks = np.round(zeno_corr_factor*np.array([ min(data_zeno['Work'].min(), data_zeno['System Heat'].min(), data_zeno['Meter Heat'].min()),\
                    max(data_zeno['Work'].max(), data_zeno['System Heat'].max(), data_zeno['Meter Heat'].max()), 0]),1)
    axs[0].set_yticks(ax_0_y_ticks)
    axs[1].set_yticks(ax_1_y_ticks)

    # Have no x-label for the first subplot
    axs[0].set_xlabel('')
    axs[1].set_xlabel(f'{symbol_dict["temperature"]}')
    plt.legend()
    # Move the legends to the right and make them not overlap
    #ax1_eff.legend().set_bbox_to_anchor((0.5, 1.05))
    plt.savefig('images/poster_plots/poster_work_heat_comparison.pdf', format='pdf', dpi=300)

# ----------------------------------------------------------------------------------------------
    # Next plot the entropy for the opt_eq_temp case
    #data_opt_eq_temp = pd.read_csv('data/params_vs_temp_opt_eq_temp.csv', skiprows=1)
    entropy_plot_data = pd.read_csv('data/params_vs_temp_zeno.csv', skiprows=1)
    zeno_corr_factor = 1
    fig, ax = plt.subplots(figsize=(12, 8))
    # First extract the system temperature T_S from the title string
    title_string = pd.read_csv('data/params_vs_temp_opt_eq_temp.csv', nrows=1)
    pattern = r"System temperature: (\d+\.\d+)"
    T_S = float(re.search(pattern, title_string.columns[0]).group(1))
    # The x-axis is T_M/T_S so x_axis* T_S = T_M
    entropy_plot_data['Meter temperature'] = entropy_plot_data['Temperature']*T_S
    # Create vectors for the system and meter heat divided by the system and meter temperature
    entropy_plot_data['System Heat / System Temperature'] = zeno_corr_factor*entropy_plot_data['System Heat']/T_S
    entropy_plot_data['Meter Heat / Meter Temperature'] = zeno_corr_factor*entropy_plot_data['Meter Heat']/entropy_plot_data['Meter temperature']
    # Next plot system heat over system temperature and meter heat over meter temperature
    sns.lineplot(x='Temperature', y='System Heat / System Temperature', data=entropy_plot_data,\
                  label=r'$\frac{Q_S}{T_S}$')
    sns.lineplot(x='Temperature', y='Meter Heat / Meter Temperature', data=entropy_plot_data,\
                    label=r'$\frac{Q_M}{T_M}$')
    # Plot the sum of system heat/system temperature and the mutual information
    entropy_plot_data['Total entropy'] = entropy_plot_data['System Heat / System Temperature'] + entropy_plot_data['Information']
    entropy_plot_data['S+I_m']= entropy_plot_data['System Heat / System Temperature'] + entropy_plot_data['Mutual Information']
    #sns.lineplot(x='Temperature', y='Total entropy', data=entropy_plot_data, label=r'$\frac{Q_S}{T_S} + I$',\
    #               color="black", linestyle='--')
    sns.lineplot(x='Temperature', y='S+I_m', data=entropy_plot_data, label=r'$\frac{Q_S}{T_S} + I_{mut}$',\
                     color="black", linestyle='dotted')
    # Add horizontal line at 0
    plt.axhline(0, color='black', linewidth=1.0)
    plt.xlabel(f'{symbol_dict["temperature"]}')
    plt.ylabel(r'Entropy [meV/K]')
    plt.legend()
    #plt.ylim(-0.02, 0.1)
    plt.tight_layout()
    sns.despine()
    plt.savefig('images/poster_plots/poster_entropy.pdf', format='pdf', dpi=300)

def W_net_per_I_tot(df_list, normalize=False, labels=None):
    """ Plot the net work extracted per unit of information for the different cases"""
    fig, ax = plt.subplots(figsize=(12, 8))
    if labels is None or len(labels) != len(df_list):
        labels = [f'Case {i}' for i in range(len(df_list))]
    for df, label in zip(df_list, labels):
        W_net_per_I = df['Work']/(df['Information'])
        if normalize:
            W_net_per_I /= W_net_per_I.max()
        sns.lineplot(x='Time', y=W_net_per_I, data=df, label=label, ax=ax)
    plt.title('Net work extracted per unit of information')
    plt.xlabel(f'{symbol_dict["time"]}')
    plt.ylabel(r'$W_{net} / I$')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/dissipation/W_net_per_I_tot.pdf', dpi=300)
    plt.close()

def information_plot(df_list, normalize=False, labels=None):
    """ Plot the information for the different cases"""
    fig, ax = plt.subplots(figsize=(12, 8))
    if labels is None or len(labels) != len(df_list):
        labels = [f'Case {i}' for i in range(len(df_list))]
    for df, label in zip(df_list, labels):
        if normalize:
            df['Information'] /= df['Information'].max()
        sns.lineplot(x='Time', y='Information', data=df, label=label, ax=ax)
    plt.title('Information')
    plt.xlabel(f'{symbol_dict["time"]}')
    plt.ylabel(f'{symbol_dict["information"]}')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/dissipation/information.pdf', dpi=300)
    plt.close()
def measurement_work_plot(df_list, normalize=False, labels=None):
    """ Plot the measurement work for the different cases"""
    fig, ax = plt.subplots(figsize=(12, 8))
    if labels is None or len(labels) != len(df_list):
        labels = [f'Case {i}' for i in range(len(df_list))]
    for df, label in zip(df_list, labels):
        if normalize:
            df['Meter Heat'] /= df['Meter Heat'].max()
        sns.lineplot(x='Time', y='Meter Heat', data=df, label=label, ax=ax)
    plt.title('Measurement work')
    plt.xlabel(f'{symbol_dict["time"]}')
    plt.ylabel(f'{symbol_dict["measurement work"]}')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/dissipation/measurement_work.pdf', dpi=300)
    plt.close()
def extracted_work_plot(df_list, normalize=False, labels=None):
    """Plot the extracted work as a function of time for the different cases

    Args:
        df_list (list): list of dataframes
        normalize (bool, optional): Normalize the data? Defaults to False.
        labels (list, optional): List of strings that label the dataframes. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    if labels is None or len(labels) != len(df_list):
        labels = [f'Case {i}' for i in range(len(df_list))]
    for df, label in zip(df_list, labels):
        if normalize:
            df['System Heat'] /= df['System Heat'].max()
        df['Extracted Work'] = -df['Meter Heat']
        sns.lineplot(x='Time', y='Extracted Work', data=df, label=label, ax=ax)
    plt.title('Extracted work')
    plt.xlabel(f'{symbol_dict["time"]}')
    plt.ylabel(f'{symbol_dict["work"]}')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/dissipation/extracted_work.pdf', dpi=300)
    plt.close()
def net_work_plot(df_list, normalize=False, labels=None):
    """Plot the net work as a function of time for the different cases

    Args:
        df_list (list): list of dataframes
        normalize (bool, optional): Normalize the data? Defaults to False.
        labels (list, optional): List of strings that label the dataframes. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    if labels is None or len(labels) != len(df_list):
        labels = [f'Case {i}' for i in range(len(df_list))]
    for df, label in zip(df_list, labels):
        if normalize:
            df['Work'] /= df['Work'].max()
        sns.lineplot(x='Time', y='Work', data=df, label=label, ax=ax)
    plt.title('Net work')
    plt.xlabel(f'{symbol_dict["time"]}')
    plt.ylabel(f'{symbol_dict["work"]}')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/dissipation/net_work.pdf', dpi=300)
    plt.close()

def zeno_crossover_plot(Q_S, x, mu_range):
    x_vals = [1,1.5,2, 0.5]
    plt.figure(figsize=(12, 8))
    for x in x_vals:
        res = np.ones(len(mu_range))
        for i,mu in enumerate(mu_range):
            # Prefactor
            #res[i] = (1-np.exp(-mu*Q_S/x))/(2*(1+np.exp(-Q_S)))*np.exp(-mu*Q_S/x)*\
            #( np.exp(-mu*Q_S/x)/(1-np.exp(-mu*Q_S/x)) +\
            # (1+np.exp(2*mu*Q_S/x)) *\
            #    ( np.exp(-mu*Q_S/x)/(1-np.exp(-mu*Q_S/x)) + np.exp(-2*mu*Q_S/x)/(1-np.exp(-mu*Q_S/x)**2) ) )        
            res[i] = 1/(1+np.exp(-Q_S))*(1-np.exp(mu*Q_S/x))**2*np.exp(-2*mu*Q_S/x)
        # Plot res against mu
        sns.lineplot(x=mu_range, y=res, linewidth=lw, label=f"$x={x}$")
    plt.xlabel(r'$\mu = \frac{\hbar\omega}{\Delta E}$')
    plt.ylabel("Zeno work condition")
    # Plot the straight line y = mu
    sns.lineplot(x=mu_range, y=mu_range, linewidth=lw, color=sns.color_palette("colorblind")[1], label=r'$\mu$')
    plt.title("Zeno work condition crossover")
    # Set y-limits defined by res
    scale = 1.2
    plt.ylim(0, max(np.abs(res))*scale)
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/poster_plots/zeno_cross_over.pdf', format='pdf', dpi=300)
    plt.close()

def zeno_crossover_intersection(Q_S_vals, x_vals, n):
    # Find the intersection point between the two lines for a few different x values
    plt.figure(figsize=(12, 8))
    for Q_S in Q_S_vals:
        mu_intersection = np.zeros(len(x_vals))
        mu_intersection_2 = np.zeros(len(x_vals))
        last_mu = 1
        for i,x in enumerate(x_vals):
            #mu_intersection[i] = sp.optimize.fsolve(lambda mu: n/(1+np.exp(-Q_S))*np.exp(-mu*Q_S/x)*4*np.cosh(mu*Q_S/(2*x))**2 -mu, 1)
            mu_intersection[i] = sp.optimize.fsolve(lambda mu: 4/(1+np.exp(-Q_S))*np.exp(-mu*Q_S/x)*np.sinh(mu*Q_S/(2*x))**2 -mu, 1)
            mu_intersection_2[i] = sp.optimize.fsolve(lambda mu: 4/(1+np.exp(-Q_S))*np.exp(-mu*Q_S/x)*np.sinh(mu*Q_S/(2*x))**2 -mu, 0.2)
            last_mu = mu_intersection[i]
        sns.lineplot(x=x_vals, y=mu_intersection, linewidth=lw, label=r"$\frac{\Delta E}{k_B T_S}$"+f"={Q_S}", color='black')
        sns.lineplot(x=x_vals, y=mu_intersection_2, linewidth=lw, color='black')
        # Fit an
    plt.xlabel(f"{symbol_dict['temperature']}")
    plt.ylabel(f"{symbol_dict['hw/de']}")
    # Color the region above the curve blue, below the curve for x=0 to 1 red
    plt.fill_between(x_vals[x_vals<=1], mu_intersection[x_vals<=1], 2, color='yellow', alpha=0.3)
    plt.fill_between(x_vals[x_vals<=1], mu_intersection[x_vals<=1], 0, color='red', alpha=0.3)
    plt.fill_between(x_vals[x_vals>=1], mu_intersection[x_vals>=1], 2, color='blue', alpha=0.3)
    # Add the text HE, IE, HP, Fridge in the lower left, lower right, upper left and upper right corners
    plt.text(0.5, 0.5, 'HE', fontsize=30, color='black')
    plt.text(1.5, 0.5, 'IE', fontsize=30, color='black')
    plt.text(0.5, 1.5, 'HP', fontsize=30, color='black')
    plt.text(1.35, 1.5, 'Fridge', fontsize=30, color='black')


    # Set lower y-limit to 0 and upper y-limit to the current upper y-limit
    #plt.ylim(0, plt.ylim()[1])
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/poster_plots/zeno_cross_over_intersection.pdf', dpi=300)
    plt.close()
def crossover_function(mu, Q_S, x):
    return 1/(1+np.exp(-Q_S))*(1-np.exp(mu*Q_S/x))**2*np.exp(-2*mu*Q_S/x)
    #return (1-np.exp(-mu*Q_S/x))/(2*(1+np.exp(-Q_S)))*np.exp(-mu*Q_S/x)*\
    #        ( np.exp(-mu*Q_S/x)/(1-np.exp(-mu*Q_S/x)) +\
    #         (1+np.exp(2*mu*Q_S/x)) *\
    #            ( np.exp(-mu*Q_S/x)/(1-np.exp(-mu*Q_S/x)) + np.exp(-2*mu*Q_S/x)/(1-np.exp(-mu*Q_S/x)**2) ) )

def plot_colored_segments(ax, x, y, condition1, condition2, color1, color2, color3):
    # Save the points that belong to the same color in a list
    color_HE = 'blue'
    color_IE = 'red'
    color_HP = 'black'
    x_HE, y_HE = [], []
    x_IE, y_IE = [], []
    x_HP, y_HP = [], []
    for i in range(len(x) -1):
        if condition1[i] and condition2[i]:
            x_HE.append(x[i])
            y_HE.append(y[i])
        elif (not condition1[i]) and condition2[i]:
            x_IE.append(x[i])
            y_IE.append(y[i])
        else:
            x_HP.append(x[i])
            y_HP.append(y[i])
    ax.plot(x_HE, y_HE, color=color_HE, linewidth=lw, label=r'$\eta_{HE}$', linestyle='--')   
    ax.plot(x_IE, y_IE, color=color_IE, linewidth=lw, label=r'$\eta_{IE}$', linestyle='--')
    ax.plot(x_HP, y_HP, color=color_HP, linewidth=lw, label=r'$COP$', linestyle='--')
    #ax.scatter(x_HE, y_HE, color=color_HE, linewidth=lw, label=r'$\eta_{HE}$', s=10)
    #ax.scatter(x_IE, y_IE, color=color_IE, linewidth=lw, label=r'$\eta_{IE}$', s=10)
    #ax.scatter(x_HP, y_HP, color=color_HP, linewidth=lw, label=r'$COP$', s=10)

def calc_efficiency(row):
    T, W, Q_S, Q_M = row['Temperature'], row['Work'], row['System Heat'], row['Meter Heat']
    if W >= 0:
        if T < 1:
            return np.abs(W)/np.abs(Q_S)
        else:
            return (np.abs(W) + np.abs(Q_S)) /np.abs(Q_M) 
    else:
        return np.abs(Q_S)/np.abs(W)

def omega_per_dE_comparison():
    x_list = [0.1, 0.5, 1.0]
    #fig, ax = plt.subplots(figsize=(12, 8))
    plt.figure(figsize=(12, 8))
    for x in x_list:
        data = pd.read_csv(f'data/omega_per_delta_E_comparison/{x}.csv', skiprows=1) 
        # Create the Q-factor
        data['Q-factor'] = data['Mutual Information'] / (data['Meter Heat'])
        data['Q-factor'] = data['Q-factor'] / data['Q-factor'].max()
        sns.lineplot(x=data['hw/dE'], y=data['Q-factor'], label=r'$T_M/T_S$'+f' = {x}')
    #ax.set_xlabel(r'$\hbar\omega/\Delta E$')
    #ax.set_ylabel(r'$Q/Q_{max}$')
    #ax.legend(loc='lower right')
    plt.xlabel(r'$\hbar\omega/\Delta E$')
    plt.ylabel(r'$Q/Q_{max}$')
    plt.legend(loc='center')
    plt.tight_layout()
    plt.savefig('images/poster_plots/omega_per_dE_comparison.pdf', format='pdf', dpi=300)
    plt.close()

def zeno_work_plot(Q_S, Q_M, x_range, n_prime):
    plt.figure(figsize=(12, 8))
    f_of_beta = n_prime*(1-np.exp(Q_M/x_range))**2*np.exp(-Q_M*(n_prime+1)/x_range)
    cut_off = Q_M*(1+np.exp(-Q_S))*Q_S
    sns.lineplot(x=x_range, y=f_of_beta, label=r'$f(T_M)$', linewidth=lw)
    plt.axhline(cut_off, color=sns.color_palette("colorblind")[1], linestyle='--', label=r'$\frac{\hbar\omega}{a\Delta E}$')
    # Find the intersection points between the two lines
    ind_1 = np.argmin(np.abs(f_of_beta[0:int(len(f_of_beta)/2)] - cut_off))
    ind_2 = np.argmin(np.abs(f_of_beta[int(len(f_of_beta)/2):] - cut_off)) + int(len(f_of_beta)/2)
    red_region = x_range[x_range < 1][ind_1:]
    plt.fill_between(red_region, f_of_beta[x_range < 1][ind_1:], cut_off, color='red', alpha=0.3)
    # Fill the region under the curve and under the cut-off in blue
    plt.fill_between(x_range, 0, f_of_beta, where=f_of_beta <cut_off, color='blue', alpha=0.3)
    plt.fill_between(x_range, 0, cut_off, where=f_of_beta >cut_off, color='blue', alpha=0.3)
    # Plot from ind_2 to the end of the array in blue
    plt.plot(x_range[ind_1], f_of_beta[ind_1], 'kx')
    plt.plot(x_range[ind_2], f_of_beta[ind_2], 'kx')

    #plt.plot(x_range, f_of_beta)
    #plt.hlines(cut_off, x_range[0], x_range[-1], color=sns.color_palette["colorblind"][1], linestyle='--', label=r'$\frac{\hbar\omega}{a\Delta E}$')
    plt.xlabel(f'{symbol_dict["temperature"]}')
    plt.ylabel(r'$f(T_M)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/poster_plots/zeno_work_plot.pdf', format='pdf', dpi=300)
    plt.close()

def zeno_crossover_intersection_2(Q_S_vals, x_vals, n):
    plt.figure(figsize=(12, 8))
    for Q_S in Q_S_vals:
        mu_intersection = np.zeros(len(x_vals))
        for i,x in enumerate(x_vals):
            mu = np.linspace(0, 5, 10000)
            #f_of_beta = n/(1+np.exp(-Q_S))*(1-np.exp(mu*Q_S/x))**2*np.exp(-mu*Q_S*(n+1)/x)
            #mu_intersection[i] = mu[np.argmin(np.abs(f_of_beta - mu))]
        sns.lineplot(x=x_vals, y=mu_intersection, linewidth=lw, label=r"$\Delta E/k_B T_S$"+f"={Q_S}")
    plt.xlabel(f"{symbol_dict['temperature']}")
    plt.ylabel(f"{symbol_dict['hw/de']}")
    # Color the region above the curve blue, below the curve for x=0 to 1 red
    plt.fill_between(x_vals, mu_intersection, max(mu_intersection), color='blue', alpha=0.3)
    plt.fill_between(x_vals[x_vals<=1], mu_intersection[x_vals<=1], 0, color='red', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/poster_plots/zeno_cross_over_intersection_2.png', dpi=300)
    plt.close()

def entropy_v2(data, fname='entropy_v2.png', title=None, T_S=300):
    plt.figure(figsize=(12, 8))
    # The x-axis is T_M/T_S so x_axis* T_S = T_M
    data['Meter temperature'] = data['Temperature']*T_S
    scale = 20.0
    # Create vectors for the system and meter heat divided by the system and meter temperature
    data['System Entropy'] = data['System Heat']/T_S
    data['Meter Entropy'] = data['Meter Heat']/data['Meter temperature']
    # Plot the "conventional" entropy S_system + S_meter
    data['Total Entropy'] = data['System Entropy'] + data['Meter Entropy']
    sns.lineplot(x='Temperature', y='Total Entropy', data=data,\
                  label=r'$S$')
    # Plot the total information, I_mutual + I_observer
    data['Total Information'] = data['Mutual Information'] + data['Observer Information']
    plt.axhline(0, color='black', linewidth=2.5)
    sns.lineplot(x='Temperature', y=data['Total Information']/scale, data=data,\
                  label=r'$I$'+f'/{scale}')
    # Plot the sum of the total entropy and the total information
    data['Total'] = (data['Total Entropy'] + data['Total Information'])/scale
    sns.lineplot(x='Temperature', y='Total', data=data,\
                  label=r'$(S + I)/$'+f'{scale}', linestyle='-.', color='black')
    plt.xlabel(f'{symbol_dict["temperature"]}')
    plt.ylabel('Entropy (a.u.)')
    plt.ylim(-0.05, 0.05)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'images/{fname}', dpi=300)

def first_law_quantities(data, fname='first_law.png', title=None, T_S=300):
    plt.figure(figsize=(12,8))
    plt.axhline(0, color='black', linewidth=2.5)
    sns.lineplot(x='Temperature', y='Work', data=data, label=r'$W$', color='grey')
    sns.lineplot(x='Temperature', y='System Heat', data=data, label=r'$Q_S$')
    sns.lineplot(x='Temperature', y='Meter Heat', data=data, label=r'$Q_M$')
    plt.xlabel(f'{symbol_dict["temperature"]}')
    plt.ylabel('Energy (a.u.)')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'images/{fname}', dpi=300)

def phase_diagram_preprocessing(main_data_names, ending_data_names=None):
    data_list = []
    for main_data_name in main_data_names:
        df = pd.read_csv(main_data_name, skiprows=1, header=None)
        df.columns = ['Temperature', 'hw/dE']
        
        if ending_data_names:
            ending_data_name = ending_data_names[main_data_names.index(main_data_name)]
            df_ending = pd.read_csv(ending_data_name, skiprows=1, header=None)
            df_ending.columns = ['Temperature', 'hw/dE']
            # Merge the two dataframes
            df = pd.concat([df, df_ending])
        
        # Sort the values by temperature
        df = df.sort_values(by='Temperature')
        # Convert the 'hw/dE' column to a list
        df['hw/dE'] = df['hw/dE'].apply(convert_to_list)
        # Drop rows where the 'hw/dE' column contains an empty list
        df = df.loc[df['hw/dE'].apply(lambda x: len(x) >0)]
        # Separate the 'hw/dE' column into two columns
        df['hw/dE_1'] = df['hw/dE'].apply(lambda x: x[-1] if len(x) > 0 else None)
        df['hw/dE_2'] = df['hw/dE'].apply(lambda x: x[0] if len(x) > 1 else None)
        data_list.append(df)
    
    return data_list

def phase_diagram(data_list, fname='phase_diagram.png', title=None, labels=None):
    # Create a figure with 4 subplots in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Do some color magic
    cmap = plt.get_cmap('seismic')
    color_points = [0.25, 0.75]
    colors = [cmap(i) for i in color_points]
    red = colors[1]
    blue = colors[0]
    cmap = plt.get_cmap('Wistia')
    color_points = [0.2]
    colors = [cmap(i) for i in color_points]
    yellow = colors[0]
    # Loop over the dataframes and plot the data in the subplots
    for i, df in enumerate(data_list):
        df = df.dropna(how='any', subset=['hw/dE_1'])
        df.loc[:, 'hw/dE_2'] = df['hw/dE_2'].fillna(0)
        sns.lineplot(x='Temperature', y='hw/dE_1', data=df, ax=axs[i//2, i%2], color='black')
        # Plot the phase boundary
        sns.lineplot(x='Temperature', y='hw/dE_2', data=df, ax=axs[i//2, i%2], color='black')
        # Change the limits of the subplot x-axis and y-axis to 0 and 2
        axs[i//2, i%2].set_xlim(0, 2)
        axs[i//2, i%2].set_ylim(0, 2)
        # Change the x-axis and y-axis labels
        axs[i//2, i%2].set_xlabel(f'{symbol_dict["temperature"]}')
        axs[i//2, i%2].set_ylabel(f'{symbol_dict["hw/de"]}')
        # If there is a label, print it in the upper right corner in a white box
        if labels is not None:
            axs[i//2, i%2].text(1.5, 1.5, labels[i], fontsize=14, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

        # Fill with blue the entire region with temperature above 1 using axvspan
        axs[i//2, i%2].axvspan(1, 2, facecolor=blue, alpha=0.7, edgecolor='none')
        # Fill the region between the two phase boundaries and below temperature 1 with red
        axs[i//2, i%2].fill_between(df['Temperature'], df['hw/dE_1'], df['hw/dE_2'], where=df['Temperature'] < 1, facecolor=red, alpha=0.7, edgecolor='none')
        # Fill the region below hw/dE_2 but above the x-axis with up to temperature 1 with yellow
        axs[i//2, i%2].fill_between(df['Temperature'], 0, df['hw/dE_2'], where=(df['Temperature'] < 1), facecolor=yellow, alpha=0.7, edgecolor='none')
        # Fill the region above hw/dE_1 below temperature 1 with yellow
        axs[i//2, i%2].fill_between(df['Temperature'], df['hw/dE_1'], 2, where=(df['Temperature'] < 1), facecolor=yellow, alpha=0.7, edgecolor='none')
        # Fill the region between the two phase boundaries and above temperature 1 with white
        axs[i//2, i%2].fill_between(df['Temperature'], df['hw/dE_1'], df['hw/dE_2'], where=df['Temperature'] > 1, facecolor='white', edgecolor='none')
        
        # For the first subplot, add some text to indicate the four regions
        if i == 0:
            axs[i//2, i%2].text(0.3, 0.5, 'HE', fontsize=30, color='black')
            axs[i//2, i%2].text(1.05, 0.5, 'IE', fontsize=30, color='black')
            axs[i//2, i%2].text(0.3, 1.2, 'HP', fontsize=30, color='black')
            axs[i//2, i%2].text(1.05, 1.2, 'RF', fontsize=30, color='black')

    # Make each row share the same y-axis and each column share the same x-axis
    for i in range(2):
        axs[i, 1].set_ylabel('')
        axs[0, i].set_xlabel('')
        axs[i, 0].set_yticks([0, 0.5, 1, 1.5, 2])
        axs[i, 1].set_yticks([0, 0.5, 1, 1.5, 2])
        axs[0, i].set_xticks([0, 0.5, 1, 1.5, 2])
        axs[1, i].set_xticks([0, 0.5, 1, 1.5, 2])
    # Remove the padding between the subplots
    plt.tight_layout()
    # Set the title of the figure
    fig.suptitle(title)
    plt.savefig(f'images/{fname}', dpi=300)

def phase_diagram_comparison(df, fname='phase_diagram_comparison.png'):
    """ Comparing the numerica solution to the analytical solution for the phase diagram"""
    df = df.dropna(how='any', subset=['hw/dE_1'])
    df.loc[:, 'hw/dE_2'] = df['hw/dE_2'].fillna(0)
    plt.figure(figsize=(12, 8))
    # Plotting the numerical solution
    sns.lineplot(x='Temperature', y='hw/dE_1', data=df, color=sns.color_palette('colorblind')[0], label='Numerical')
    sns.lineplot(x='Temperature', y='hw/dE_2', data=df, color=sns.color_palette('colorblind')[0])
    # Plotting the analytical solution
    Q_S = 4.33
    x_vals = np.linspace(0, 1.75, 10000)
    mu_intersection = np.zeros(len(x_vals))
    mu_intersection_2 = np.zeros(len(x_vals))
    mu_1 = 0.99
    mu_2 = 0.01
    a1 = 0.49
    b1 = 1.0
    for i,x in enumerate(x_vals):
        try:
            # Find the first root in the interval [a1, b1]
            mu_intersection[i] = sp.optimize.brentq(zeno_function, a1, b1, args=(x,))
            
        except ValueError as e:
            print(f"Failed to find root at x={x}: {e}")
            mu_intersection[i] = np.nan
        try:
            # Find the second root in the interval [a2, b2]
            #mu_intersection_2[i] = sp.optimize.fsolve(lambda mu: 4/(1+np.exp(-Q_S))*np.exp(-mu*Q_S/x)*np.sinh(mu*Q_S/(2*x))**2 -mu, mu_2)
            mu_intersection_2[i] = sp.optimize.brentq(zeno_function, 1e-8, 0.49, args=(x,))
            mu_2 = mu_intersection_2[i]
        except ValueError as e:
            print(f"Failed to find root at x={x}: {e}")
            mu_intersection_2[i] = np.nan
    sns.lineplot(x=x_vals, y=mu_intersection, color=sns.color_palette('colorblind')[1], label='Analytical', linestyle='--')
    sns.lineplot(x=x_vals, y=mu_intersection_2, color=sns.color_palette('colorblind')[1], linestyle='--')
    plt.xlabel(f'{symbol_dict["temperature"]}')
    plt.ylabel(f'{symbol_dict["hw/de"]}')
    plt.legend()#
#
#
#
#
    plt.tight_layout()
    plt.savefig(f'images/{fname}', dpi=300)
def boundary_one(mu, x):
    Q_S = 4.33
    return (mu*Q_S/x) + np.log( 1+np.sqrt((1+np.exp(-Q_S))*mu) )
def boundary_two(mu, x):
    Q_S = 4.33
    if mu < 1/(1+np.exp(-Q_S)):
        return (mu*Q_S/x) + np.log( 1-np.sqrt((1+np.exp(-Q_S))*mu) )
    else:
        return 0

def power_heatmap(indata, times, fname='power_plot.png', title=None, labels=None, overlay=False, xlim=None, ylim=None):
    """Takes work-data and time-data and plots the power as a heatmap

    Args:
        data (list): List of dataframes with the work data
        times (list): The times corresponding to the work data
        fname (str, optional): File name to save the plot. Defaults to 'power_plot.png'.
        title (str, optional): Title of the plot if desired. Defaults to None.
        labels (list/str, optional): Labels to set on each of the plots. Defaults to None.
        xlim (float, optional): Limits the x-values by filtering the data. Defaults to None.
        ylim (float, optional): Limits the y-values by filtering the data. Defaults to None.
    """
    # Create a custom colormap
    global_max_val = 0
    if not isinstance(indata, list):
        indata = [indata]
    for i, df in enumerate(indata):
        global_max_val = max(global_max_val, df.max().max()/times[i])
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, df in enumerate(indata):
        df = df[df.columns[df.columns.astype(float) < xlim]]
        df = df[df.index.astype(float) < ylim]
        df.columns = df.columns.astype(float)
        df.index = df.index.astype(float)
        #if df.max().max()/np.float64(times[i]) < global_max_val*1e-2:
        #    r = 50
        #    cmap = plt.get_cmap('seismic')
        #    colors = cmap(np.linspace(0.4, 0.6, 256))
        #    cmap = mcolors.LinearSegmentedColormap.from_list('custom_seismic', colors)
        #    norm = mcolors.TwoSlopeNorm(vmin=-df.max().max()/times[i]/r, vcenter=0, vmax=df.max().max()/times[i]/r)
        #else:
        #    norm =  mcolors.TwoSlopeNorm(vmin=df.min().min(), vcenter=0, vmax=df.max().max())
        #    cmap = plt.get_cmap('seismic')
        norm = mcolors.TwoSlopeNorm(vmin=df.min().min()/np.float64(times[i]), vcenter=0, vmax=df.max().max()/np.float64(times[i]))
        cmap = plt.get_cmap('seismic')
        sns.heatmap(df/np.float64(times[i]), ax=axs[i//2, i%2], cmap=cmap, cbar_kws={'label': 'Power [meV/s]'}, norm=norm)
        n_cols = len(df.columns)
        n_rows = len(df.index)
        axs[i//2, i%2].set_xticks(np.linspace(0, n_cols, 5))
        axs[i//2, i%2].set_yticks(np.linspace(0, n_rows, 5))
        xticklabels = np.linspace(df.columns[0], df.columns[-1], 5)
        xticklabels = [f'{x:.2f}' for x in xticklabels]
        yticklabels = np.linspace(df.index[0], df.index[-1], 5)
        yticklabels = [f'{y:.2f}' for y in yticklabels]
        axs[i//2, i%2].set_xticklabels(xticklabels)
        axs[i//2, i%2].set_yticklabels(yticklabels)
        axs[i//2, i%2].invert_yaxis()
        axs[i//2, i%2].set_xlabel(f'{symbol_dict["temperature"]}')
        axs[i//2, i%2].set_ylabel(f'{symbol_dict["hw/de"]}')
        if overlay:
            add_overlay(axs[i//2, i%2], df, times[i])
        
    plt.tight_layout()
    if title:
        fig.suptitle(title)
    if overlay:
        # If overlay is true then split fname at the dot and add _overlay before the dot
        fname = fname.split('.')
        fname = fname[0] + '_overlay.' + fname[1]
    plt.savefig(f'images/{fname}', dpi=300)

def zeno_function(mu,x):
    Q_S = 4.33
    return (4/(1+np.exp(-Q_S))*np.exp(-mu*Q_S/x)*np.sinh(mu*Q_S/(2*x))**2 -mu)

def multidata_preprocessing(filenames):
        # Ensure filenames is a list
    if isinstance(filenames, str):
        filenames = [filenames]
    
    df_lists = []  # To store lists of dataframes for each file

    for filename in filenames:
        # Open the file and read it line by line
        with open(filename, 'r') as file:
            # Skip the first line (supplementary information)
            file.readline()
            
            # Extract x-values from the header (second line)
            x_values = file.readline().strip().split(',')[0:-1]  # Skip the last element (empty string)

            # Initialize empty lists for data
            u_values = []
            W_data = []
            We_data = []
            Wm_data = []

            # Process each remaining line
            for line in file:
                # Clean and split the line into individual list elements
                row = line.strip()[1:-2]  # Remove outer brackets
                row = re.sub(r'np\.float64\((.*?)\)', r'\1', row)  # Replace np.float64
                cleaned_row = row.split('],[')  # Split into individual cells

                # Process the first element to extract the u value
                first_cell = ast.literal_eval(f"[{cleaned_row[0]}]")  # Ensure valid list syntax
                u_values.append(first_cell[0])  # Extract the u value

                # Process the rest of the row
                W_row, We_row, Wm_row = [], [], []
                for cell in cleaned_row:
                    values = ast.literal_eval(f"[{cell}]")  # Convert string to list
                    W_row.append(values[1])
                    We_row.append(values[2])
                    Wm_row.append(values[3])
                
                # Append rows to respective lists
                W_data.append(W_row)
                We_data.append(We_row)
                Wm_data.append(Wm_row)

        # Create DataFrames
        df_W = pd.DataFrame(W_data, index=u_values, columns=x_values)
        df_We = pd.DataFrame(We_data, index=u_values, columns=x_values)
        df_Wm = pd.DataFrame(Wm_data, index=u_values, columns=x_values)

        # Append the list of dataframes for this file
        df_lists.append([df_W, df_We, df_Wm])

    # Return based on the number of files
    if len(df_lists) == 1:
        return df_lists[0]  # Return a single list of dataframes for one file
    else:
        return df_lists  # Return a list of lists of dataframes for multiple files

def phase_diagram_2(data_list, fname='phase_diagram_2.png', title=None, labels=None):
    # Phase diagram creator for the multidata files
    for df in data_list:
        df.columns = df.columns.astype(float)
        df.index = df.index.astype(float)
        # Look at column in the dataframe and find the indices where the values change sign
        # This is the phase boundary, so then we save each of the points (column_name, index1), (column_name, index2) in a list
        phase_boundaries = []
        for column in df.columns:
            # Find the indices where the values change sign
            indices = np.where(np.diff(np.sign(df[column])))[0]
            for index in indices:
                phase_boundaries.append((column, index))
        # Create a scatter plot with the phase boundaries
        plt.figure(figsize=(12, 8))
        plt.scatter([x[0] for x in phase_boundaries], [x[1] for x in phase_boundaries], color='black')
        plt.xlabel(f'{symbol_dict["temperature"]}')
        plt.ylabel(f'{symbol_dict["hw/de"]}')
        plt.title(title)
        plt.tight_layout()
    plt.show()
def add_overlay(ax, df, time):
    # Add an overlay of the phase boundaries
    phase_boundaries=[]
    for column in df.columns:
        # Find the indices where the values change sign
        indices = np.where(np.diff(np.sign(df[column])))[0]
        for index in indices:
            phase_boundaries.append((column, index+1))
    # Create a scatter plot with the phase boundaries
    ax.scatter([x[0]*500 for x in phase_boundaries], [x[1] for x in phase_boundaries], color='black', s=1)

    # Add the time in the upper right corner
    ax.text(0.9, 0.9, fr'$\tau$={time}', fontsize=14, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'), transform=ax.transAxes, ha='right', va='top')
    #ax.text(1.5, 1.5, f'time={time}', fontsize=14, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

def plot_data_broken_x_axis(df_list, xaxis, yaxis, interval_1, interval_2, fname='broken_x_axis.png', title=None, labels=None, xlabel=None, ylabel=None, legend_pos=None):
    """Creates a plot with the x-axis broken into two intervals. Uses seaborn to plot df[f'{xaxis}'] against df[f'{yaxis}'] for each dataframe in df_list. 
    The x-axis is broken into two intervals defined by interval_1 and interval_2.

    Args:
        df_list (list of pandas.dataframe): List of pandas dataframes to plot.
        xaxis (str): The quantity to plot on the x-axis.
        yaxis (str): The quantity to plot on the y-axis.
        interval_1 (tuple, list, numpy.ndarray): Interval of the left subplot. Should be a tuple, list or numpy.ndarray with two elements. Chooses the first and last element in the list or array.
        interval_2 (tuple, list, numpy.ndarray): Interval of the right subplot. Should be a tuple, list or numpy.ndarray with two elements. Chooses the first and last element in the list or array.
        fname (str, optional): Filename to save the figure, will save to f'images/{fname}'. Defaults to 'broken_x_axis.png'.
        title (str, optional): Title of the plot. Defaults to None.
        labels (list of str, optional): Labels to be added to the legend. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
        legend_pos (tuple, str, optional): Position of the legend. Either takes a str in which case 'loc' is used or takes a tuple, in which case 'anchor_to_bbox' is used.  Defaults to None.
    """
    # Select only the values where Time < 2 and Time > 98 and plot those
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    fig.subplots_adjust(wspace=0.05)
    #filter_df = df_gamma_0[(df_gamma_0['Time'] < 2) | (df_gamma_0['Time'] > 98)]
    for ax in (ax1, ax2):
        for i,df in enumerate(df_list):
            if labels is not None:
                sns.lineplot(x=xaxis, y=yaxis, data=df, ax=ax, label=labels[i])
            else:
                sns.lineplot(x=xaxis, y=yaxis, data=df, ax=ax)
    # Set the limits of the x-axes
    ax1.set_xlim(interval_1[0], interval_1[1])
    ax2.set_xlim(interval_2[0], interval_2[1])
    # Set the title of the figure
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    # Remove the spines between the two axes
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # Remove the labels on the y-axis
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    # Remove the labels on the x-axis
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    # Remove the legend from the first axis
    ax1.legend().remove()
    if isinstance(legend_pos, tuple):
        ax2.legend(bbox_to_anchor=legend_pos)
    elif isinstance(legend_pos, str):
        ax2.legend(loc=legend_pos)
    else:
        ax2.legend()
    # Set the x- and y-axis labels
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    # Add the slashes to indicate the broken x-axis
    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([1, 1], [1, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
    # Save the figure
    plt.savefig(f'images/{fname}', dpi=300)

def plot_broken_y_axis(df_list, xaxis, yaxis, interval_1, interval_2, fname='broken_y_axis.png', title=None, labels=None, xlabel=None, ylabel=None, legend_pos=None):
    """Creates a plot with the y-axis broken into two intervals. Uses seaborn to plot df[f'{xaxis}'] against df[f'{yaxis}'] for each dataframe in df_list.
    The y-axis is broken into two intervals defined by interval_1 and interval_2.

    Args:
        df_list (list of pandas.DataFrame): List of pandas dataframes to plot.
        xaxis (str): The quantity to plot on the x-axis.
        yaxis (str): The quantity to plot on the y-axis.
        interval_1 (tuple, list, numpy.ndarray): Interval of the upper subplot. Should be a tuple, list or numpy.ndarray with two elements. Chooses the first and last element in the list or array.
        interval_2 (tuple, list, numpy.ndarray): Interval of the lower subplot. Should be a tuple, list or numpy.ndarray with two elements. Chooses the first and last element in the list or array.
        fname (str, optional): Filename to save the figure. Will save to f'images/{fname}'. Defaults to 'broken_y_axis.png'.
        title (str, optional): Title of the figure. Defaults to None.
        labels (list of str, optional): Labels to be added to the legend. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
        legend_pos (tuple, str, optional): Position of the legend. Either takes a str in which case 'loc' is used or takes a tuple, in which case 'anchor_to_bbox' is used. Defaults to None.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    if not isinstance(df_list, list):
        df_list = [df_list]
    if not isinstance(yaxis, list):
        yaxis = [yaxis]
    # Loop over the dataframes and plot the data in the subplots
    for df in df_list:
        for i, y in enumerate(yaxis):
            if labels is not None:
                sns.lineplot(x=xaxis, y=y, data=df, ax=ax1, label=labels[i])
                sns.lineplot(x=xaxis, y=y, data=df, ax=ax2, label=labels[i])
            else:
                sns.lineplot(x=xaxis, y=y, data=df, ax=ax1)
                sns.lineplot(x=xaxis, y=y, data=df, ax=ax2)
    # Set the limits of the y-axes
    ax1.set_ylim(interval_1[0], interval_1[1])
    ax2.set_ylim(interval_2[0], interval_2[1])
    # Set the title of the figure
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()

    # Display ticks on the y-axis and the lower x-axis
    ax2.tick_params(axis='x', bottom=True)
    ax1.set_yticks(ax1.get_yticks()[1:]) # Remove the first y-tick since it interferes with the diagonal line for the broken axis
    ax1.tick_params(axis='y', left=True)
    ax2.tick_params(axis='y', left=True)


    # Remove the spines between the two axes
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Remove the labels on the x-axis
    ax1.set_xlabel('')
    ax2.set_xlabel('')

    # Remove the labels on the y-axis
    ax1.set_ylabel('')
    ax2.set_ylabel('')

    # Remove the legend from the first axis
    ax1.legend().remove()
    if isinstance(legend_pos, tuple):
        ax2.legend(loc='center', bbox_to_anchor=legend_pos)
    elif isinstance(legend_pos, str):
        ax2.legend(loc=legend_pos)
    else:
        ax2.legend()

    # Set the x- and y-axis labels
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    # Add the slashes to indicate the broken y-axis
    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    # Change the spacing between the subplots
    plt.subplots_adjust(hspace=0.05)

    # Save the figure
    plt.savefig(f'images/{fname}', dpi=300)

def plot_data(df_list, xaxis, yaxis, fname='plot.png', title=None, labels=None, xlabel=None, ylabel=None, legend_pos=None, xlim=None, ylim=None):
    """Creates a plot with seaborn using df[f'{xaxis}'] against df[f'{yaxis}'] for each dataframe in df_list.

    Args:
        df_list (list of pandas.dataframe): List of pandas dataframes to plot.
        xaxis (str): The quantity to plot on the x-axis.
        yaxis (str): The quantity to plot on the y-axis.
        fname (str, optional): Filename to save the figure, will save to f'images/{fname}'. Defaults to 'plot.png'.
        title (str, optional): Title of the plot. Defaults to None.
        labels (list of str, optional): Labels to be added to the legend. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
        legend_pos (tuple, str, optional): Position of the legend. Either takes a str in which case 'loc' is used or takes a tuple, in which case 'anchor_to_bbox' is used.  Defaults to None.
        xlim (tuple, list, optional): Limits for the x-axis. Defaults to None.
        ylim (tuple, list, optional): Limits for the y-axis. Defaults to None.
    """
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]
    if isinstance(labels, str):
        labels = [labels]
    plt.figure(figsize=(12, 8))
    for i,df in enumerate(df_list):
        if labels is not None:
            sns.lineplot(x=xaxis, y=yaxis, data=df, label=labels[i])
        else:
            sns.lineplot(x=xaxis, y=yaxis, data=df)
    if title is not None:
        plt.title(title)
    if isinstance(legend_pos, tuple):
        plt.legend(loc='upper right', bbox_to_anchor=legend_pos)
    elif isinstance(legend_pos, str):
        plt.legend(loc=legend_pos)
    else:
        plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.hlines(0, 0, 2, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'images/{fname}', dpi=300)
    plt.close()

def plot_multidata(df_list, xaxis, yaxis_list, fname=None, title=None, labels=None, xlabel=None, ylabel=None, legend_pos=None, xlim=None, ylim=None):
    """Function to generate a plot with seaborn for multiple dataframes and multiple quantities on the y-axis.

    Args:
        df_list (list of pandas.DataFrame): List of the dataframes to plot.
        xaxis (str): The quantity to plot on the x-axis.
        yaxis_list (list of str): The quantities to plot on the y-axis.
        fname (str, optional): The filename to save the plot to, saves to f'images/{fname}'. Defaults to None. If None, the figure is returned as an axis object.
        title (str, optional): Title of the figure. Defaults to None.
        labels (list o list of str, optional): Each element should be a list of legend labels matched to the entries in the corresponding dataframe. Defaults to None.
        xlabel (str, optional): The label on the x-axis. Defaults to None.
        ylabel (str, optional): The label on the y-axis. Defaults to None.
        legend_pos (tuple, str, optional): Position of the legend. Either takes a str un which case 'loc' is used or takes a tuple, in which case 'anchor_to_bbox' is used. Defaults to None.
        xlim (tuple, list, optional): Limits for the x-axis. Defaults to None.
        ylim (tuple, list, optional): Limits for the y-axis. Defaults to None.
    """
    # Check if df_list is a single dataframe and convert to a list if it is
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]
    # Check if labels is a single string and convert to a list if it is
    if isinstance(labels, str):
        labels = [labels]
    # Create the figure
    plt.figure(figsize=(12, 8))
    # Loop over the dataframes and the y-axis quantities
    for i,df in enumerate(df_list):
        for j,yaxis in enumerate(yaxis_list):
            if labels is not None:
                sns.lineplot(x=xaxis, y=yaxis, data=df, label=labels[i][j])
            else:
                sns.lineplot(x=xaxis, y=yaxis, data=df)
    # Add the title
    if title is not None:
        plt.title(title)
    # Add the legend
    if isinstance(legend_pos, tuple):
        plt.legend(loc='upper right', bbox_to_anchor=legend_pos)
    elif isinstance(legend_pos, str):
        plt.legend(loc=legend_pos)
    else:
        plt.legend()
    # Add the x-axis label
    if xlabel is not None:
        plt.xlabel(xlabel)
    # Add the y-axis label
    if ylabel is not None:
        plt.ylabel(ylabel)
    # Set the x-axis limits
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    # Set the y-axis limits
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    # Save the figure
    plt.tight_layout()
    if fname is not None:
        plt.savefig(f'images/{fname}', dpi=300)
        plt.close()
    else:
        return plt.gca()

def fix_broken_csv(input_file, output_file):
    """Fixes a broken csv file where the error is a premature line break. I.e.
    x,[y1 y2] has become
    x,[y1
    y2]

    Args:
        input_file (str): Path to the broken CSV file
        output_file (str): Path to the fixed CSV file
    """
    fixed_lines = []
    with open(input_file, 'r') as infile:
        buffer = "" # Buffer to store the incomplete line
        for line in infile:
            stripped_line = line.strip() # Get rid of leading/trailing whitespace
            # If buffer is empty, start processing the new line
            if not buffer:
                buffer = stripped_line
            else:
                # If buffer is not empty, append the stripped line to the buffer
                buffer += " " + stripped_line
            # If the buffer ends with a closing bracket, the line is complete
            if buffer.endswith(']'):
                fixed_lines.append(buffer)
                buffer = "" # Reset the buffer
        # If the last line is incomplete, add it to the fixed lines anyway
        if buffer:
            fixed_lines.append(buffer)
    # Write the fixed lines to the output file
    with open(output_file, 'w') as outfile:
        for line in fixed_lines:
            outfile.write(line + '\n')
def file_fixer():
    # Just here so that I can save the code somewhere if I need to reuse it
    taus = [1e-06, 0.125, 0.25, 0.5]
    unitary_indata = [f'data/phase_boundary_opt_eq_temp_tau={tau}_ending.csv' for tau in taus]
    unitary_outdata = [f'data/phase_boundary_unitary_tau={tau}_ending.csv' for tau in taus]
    dissipation_indata = [f'data/phase_boundary_tau={tau}_gamma=0.01_ending.csv' for tau in taus]
    dissipation_outdata = [f'data/phase_boundary_dissipation_tau={tau}_ending.csv' for tau in taus]
    for unit_in, unit_out, diss_in, diss_out in zip(unitary_indata, unitary_outdata, dissipation_indata, dissipation_outdata):
        fix_broken_csv(unit_in, unit_out)
        fix_broken_csv(diss_in, diss_out)
    # Create a new folder called broken_csv_file_backup and move the broken csv files there
    if not os.path.exists('broken_csv_file_backup'):
        os.makedirs('broken_csv_file_backup')
    for unit_in, diss_in in zip(unitary_indata, dissipation_indata):
        shutil.move(unit_in, 'broken_csv_file_backup')
        shutil.move(diss_in, 'broken_csv_file_backup')

def efficiency_plot(df, fname='images/efficiency_plot.png'):
    """The efficiency is a bit wonky so it gets its own plot function.

    Args:
        df (pandas.DataFrame): The dataframe to plot the data from
    """
    # First plot the efficiency in the temp < 1, work > 0 region
    # Then plot the efficiency in the temp < 1, work < 0 region
    # Then plot the efficiency in the temp > 1 region, work > 0 if it exists
    # Then plot the efficiency in the temp > 1, work < 0 region
    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot the efficiency in the temp < 1, work > 0 region, i.e. the HE region
    sns.lineplot(x='Temperature', y='Efficiency', data=df[(df['Temperature'] < 1) & (df['Work'] > 0)], label=r'$\eta_{HE}$')
    # Plot the efficiency in the temp < 1, work < 0 region, i.e. the HP region
    sns.lineplot(x='Temperature', y='Efficiency', data=df[(df['Temperature'] < 1) & (df['Work'] < 0)], label=r'$COP_{HP}$')
    # Plot the efficiency in the temp > 1, work > 0 region, i.e. the IE region
    sns.lineplot(x='Temperature', y='Efficiency', data=df[(df['Temperature'] > 1) & (df['Work'] > 0)], label=r'$\eta_{IE}$')
    # Plot the efficiency in the temp > 1, work < 0 region, i.e. the RF region  
    sns.lineplot(x='Temperature', y='Efficiency', data=df[(df['Temperature'] > 1) & (df['Work'] < 0)], label=r'$COP_{RF}$')
    # Add vertical lines to indicate the phase boundaries where the HE region ends and where the IE region starts
    # Find where the HE starts and ends
    HE_start = df[(df['Temperature'] < 1) & (df['Work'] > 0)].index[0]
    HE_end = df[(df['Temperature'] < 1) & (df['Work'] > 0)].index[-1]
    # Find where the IE starts
    IE_start = df[(df['Temperature'] > 1) & (df['Work'] > 0)].index[0]
    # Add the vertical lines
    plt.axvline(HE_start, color='black', linestyle='--', label='HE start')
    plt.axvline(HE_end, color='black', linestyle='--', label='HE end')
    plt.axvline(IE_start, color='black', linestyle='--', label='IE start')
    # Add the legend
    plt.legend()
    # Set the x- and y-axis labels
    plt.xlabel(f'{symbol_dict["temperature"]}')
    plt.ylabel('Efficiency')
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{fname}', dpi=300)




if __name__ == "__main__":
    main()