import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import os
import re
import numpy as np

# Set the style of the plots
sns.set_style(style='white')
sns.set_context(context='poster')
sns.set_palette(palette='colorblind')
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
    #data = pd.read_csv('data/params_vs_temp_zeno.csv', skiprows=1)
    #fig, ax = plt.subplots(figsize=(12, 8))
    #sns.lineplot(x='Temperature', y='Work', data=data, label=symbol_dict['observer information'], color=color_dict['Observer Information'], linewidth=lw)
    #sns.lineplot(x='Temperature', y='System Heat', data=data, label=symbol_dict['system heat'], color=color_dict['System Heat'], linewidth=lw)
    #sns.lineplot(x='Temperature', y='Meter Heat', data=data, label=symbol_dict['meter heat'], color=color_dict['Meter Heat'], linewidth=lw)
    #plt.axhline(0, color='black',  linestyle='--')
    #plt.xlabel(f'{symbol_dict["temperature"]}')
    #plt.ylabel(f'{symbol_dict["work"]}')
    #sns.despine()
    #plt.tight_layout()
    #plt.savefig('images/obs_info_test.png')
    poster_plotting()
    zeno_crossover_plot(Q_S=4.33, x=1, mu_range=np.linspace(1e-5, 2, 100))

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

def plot_work_comparison(data, file_ending, sim_run, x_axis, title_string):
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
        plt.savefig(f'images/param_vs_{file_ending}/work__comparison_{sim_run}.png')
        plt.close()

def plot_zeno_crossover():
    data = 'data/zeno_cross_over_zeno.csv'
    data = pd.read_csv(data, skiprows=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x='Temperature Ratio', y='Work condition', data=data,\
                  label=r'$f(T_M)$', color=color_dict['Work'], linewidth=lw)
    sns.lineplot(x='Temperature Ratio', y='Constant limit', data=data,\
                  label=r'$\frac{2\hbar\omega}{a\Delta E}$', color=color_dict['Measurement Work'], \
                    linewidth=lw)
    # Color the region 0 to 1 in the x-axis red and 1 to the maximum temperature ratio blue
    plt.axvspan(0, 1, color='red', alpha=0.3)
    plt.axvspan(1, data['Temperature Ratio'].max(), color='blue', alpha=0.3)
    plt.legend()
    plt.title('Zeno Limit Cross Over')
    plt.xlabel(f'{symbol_dict["temperature"]}')
    plt.ylabel('Condition for net positive work')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'images/zeno_cross_over.png')
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
    plt.ylim(-0.01, 0.02)
    plt.title(title_string)
    plt.xlabel(f'{symbol_dict[x_axis.lower()]}')
    plt.ylabel('Entropy [meV/K]')
    plt.legend()
    sns.despine()
    plt.savefig(f'images/param_vs_{file_ending}/entropy_{sim_run}.png')
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
    data_opt = pd.read_csv('data/params_vs_time_opt.csv', skiprows=1)
    data_opt_eq_temp = pd.read_csv('data/params_vs_time_opt_eq_temp.csv', skiprows=1)
    data_zeno_eq_temp = pd.read_csv('data/params_vs_time_zeno_eq_temp.csv', skiprows=1)
    color1 = sns.color_palette("colorblind")[0]
    color2 = sns.color_palette("colorblind")[1]
    color3 = sns.color_palette("colorblind")[2]


    # Plot information and work for the three different cases with the y-axis for information
    # on the left and the y-axis for work on the right
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    sns.lineplot(x='Time', y='Information', data=data_opt, label=r'$I(T_M\ll T_S)$', ax=ax1, color=color1)
    sns.lineplot(x='Time', y='Work', data=data_opt, label=r'$W_{meas}(T_M \ll T_S)$', ax=ax2, color=color1,  linestyle='--')
    sns.lineplot(x='Time', y='Information', data=data_opt_eq_temp, label=f'$I(T_M=T_S)$', ax=ax1, color=color2)
    sns.lineplot(x='Time', y='Work', data=data_opt_eq_temp, label=r'$W_{meas}(T_M=T_S)$', ax=ax2, color=color2,  linestyle='--')
    ax1.set_ylabel(f'{symbol_dict["information"]}')
    ax2.set_ylabel(f'{symbol_dict["work"]}')
    ax1.set_xlabel(f'{symbol_dict["time"]}')
    fig.tight_layout()
    ax2.legend().set_bbox_to_anchor((0.52, 0.1))
    legend1 = ax1.legend()
    legend1.set_bbox_to_anchor((0.435, 0.3))
    #ax1.legend(loc='upper left')
    #ax2.legend(loc='upper right')
    sns.despine()
    plt.savefig('images/poster_plots/poster_info_work.pdf', format='pdf', dpi=300)
    plt.close()
    fig,ax = plt.subplots(figsize=(12, 8))
    info_per_Wmsmt_opt = (data_opt['Information']/(data_opt['Meter Heat']))[1:-1]
    info_per_Wmsmt_opt /= info_per_Wmsmt_opt.max()
    info_per_Wmsm_opt_eq_temp = (data_opt_eq_temp['Information']/(data_opt_eq_temp['Meter Heat']))[1:-1]
    info_per_Wmsm_opt_eq_temp /= info_per_Wmsm_opt_eq_temp.max()
    sns.lineplot(x='Time', y=info_per_Wmsmt_opt, data=data_opt, label=r'$\frac{I}{W_{meas}}(T_M\ll T_S)$', color=color1)
    sns.lineplot(x='Time', y=info_per_Wmsm_opt_eq_temp, data=data_opt_eq_temp, label=r'$\frac{I}{W_{meas}}(T_M=T_S)$', color=color2)
    plt.title('Information per Measurement Work')
    plt.xlabel(f'{symbol_dict["time"]}')
    plt.ylabel(f'{symbol_dict["information"]} / {symbol_dict["measurement work"]}')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig('images/poster_plots/poster_info_per_Wmsmt.pdf', format='pdf', dpi=300)
    plt.close()

    # Next plot Q-factor I_mutual/W_measurement for all but the zeno case
    data_opt = pd.read_csv('data/params_vs_omega_per_delta_E_opt.csv', skiprows=1)
    data_opt_eq_temp = pd.read_csv('data/params_vs_omega_per_delta_E_opt_eq_temp.csv', skiprows=1)
    #data_big_temp = pd.read_csv('data/params_vs_omega_per_delta_E_big_temp.csv', skiprows=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    # Add to the dataframe the Q-factor I_mutual/W_measurement
    data_opt['Q-factor'] = data_opt['Mutual Information'] / (data_opt['Meter Heat'])
    Q_fact = data_opt['Q-factor'] / data_opt['Q-factor'].max()
    data_opt_eq_temp['Q-factor'] = data_opt_eq_temp['Mutual Information'] / (data_opt_eq_temp['Meter Heat'])
    Q_fact_eq_temp = data_opt_eq_temp['Q-factor'] / data_opt_eq_temp['Q-factor'].max()
    #data_big_temp['Q-factor'] = (data_big_temp['Mutual Information'] / (data_big_temp['Meter Heat']))[1:-1]
    sns.lineplot(x='hw/dE', y=Q_fact, data=data_opt, label=r'$Q(T_M\ll T_S)$', color=color1)
    sns.lineplot(x='hw/dE', y=Q_fact_eq_temp, data=data_opt_eq_temp, label=r'$Q(T_M=T_S)$', color=color2)
    #sns.lineplot(x='hw/dE', y='Q-factor', data=data_big_temp, label=r'$Q(T_M\gg T_S)$', color=color3)
    plt.title('Normalized Q-factor')
    plt.xlabel(f'{symbol_dict["hw/de"]}')
    plt.ylabel(f'{symbol_dict["q-factor"]}'+r'$/Q_{max}$')
    plt.legend()
    plt.tight_layout()
    sns.despine()
    plt.savefig('images/poster_plots/poster_q_factor.pdf', format='pdf', dpi=300)
    plt.close()

    # Next plot work and heat comparisons for three different cases, opt, opt_eq_temp and zeno_eq_temp
    data_opt = pd.read_csv('data/params_vs_temp_opt_eq_temp.csv', skiprows=1)
    data_zeno = pd.read_csv('data/params_vs_temp_zeno_eq_temp.csv', skiprows=1)
    #data_zeno = pd.read_csv('data/params_vs_temp_zeno_eq_temp.csv', skiprows=1)
    zeno_corr_factor = 1e17
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
    ax1_eff = axs[1].twinx()
    sns.lineplot(x='Temperature', y='Efficiency', data=data_opt, label=r"$\eta$", color='black', linestyle='--', ax=ax0_eff)
    sns.lineplot(x='Temperature', y='Efficiency', data=data_zeno, label=r"$\eta^{Zeno}$", color='black', linestyle='--', ax=ax1_eff)
    
    # Set y-axis for the efficiency axes
    ax0_eff.set_ylabel(r'$\eta$')
    ax1_eff.set_ylabel(r'$\eta^{Zeno}$')
    ax0_eff.set_ylim(0, 0.1*data_opt['Efficiency'].max())
    ax1_eff.set_ylim(0, 0.1*data_zeno['Efficiency'].max())

    # Color the region 0 to 1 in the x-axis red and 1 to the maximum temperature ratio blue for both subplots
    axs[0].axvspan(0, 1, color='red', alpha=0.3)
    axs[0].axvspan(1, 2, color='blue', alpha=0.3)
    axs[1].axvspan(0, 1, color='red', alpha=0.3)
    axs[1].axvspan(1, 2, color='blue', alpha=0.3)
    axs[0].axhline(0, color='black', linewidth=lw)
    axs[1].axhline(0, color='black', linewidth=lw)
    # Make them share the same x-axis
    plt.setp(axs[0].get_xticklabels(), visible=False)

    # For both axes set y-tick
    ax_0_y_ticks = np.round(np.array([min(data_opt['Work'].min(), data_opt['System Heat'].min(), data_opt['Meter Heat'].min()),\
                    max(data_opt['Work'].max(), data_opt['System Heat'].max(), data_opt['Meter Heat'].max()), 0]),1)
    ax_1_y_ticks = np.round(zeno_corr_factor*np.array([ min(data_zeno['Work'].min(), data_zeno['System Heat'].min(), data_zeno['Meter Heat'].min()),\
                    max(data_zeno['Work'].max(), data_zeno['System Heat'].max(), data_zeno['Meter Heat'].max()), 0]),1)
    axs[0].set_yticks(ax_0_y_ticks)
    axs[1].set_yticks(ax_1_y_ticks)
    axs[0].set_ylabel('Work [meV]')
    axs[1].set_ylabel(r'Work [$10^{-17}$meV]')

    # Have no x-label for the first subplot
    axs[0].set_xlabel('')
    axs[1].set_xlabel(f'{symbol_dict["temperature"]}')
    plt.tight_layout()
    plt.legend()
    # Move the legends to the right and make them not overlap
    ax1_eff.legend().set_bbox_to_anchor((0.5, 1.05))

    #sns.despine()
    plt.savefig('images/poster_plots/poster_work_heat_comparison.pdf', format='pdf', dpi=300)

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

def zeno_crossover_plot(Q_S, x, mu_range):
    Q_S = 1.0
    x_vals = [1,1.5,2, 0.5]
    plt.figure(figsize=(12, 8))
    for x in x_vals:
        res = np.ones(len(mu_range))
        for i,mu in enumerate(mu_range):
            # Prefactor
            res[i] = (1-np.exp(-mu*Q_S/x))/(1+np.exp(-Q_S))*np.exp(-mu*Q_S/x)*\
            ( np.exp(-mu*Q_S/x)/(1-np.exp(-mu*Q_S/x)) +\
             (1+np.exp(2*mu*Q_S/x)) *\
                ( np.exp(-mu*Q_S/x)/(1-np.exp(-mu*Q_S/x)) + np.exp(-2*mu*Q_S/x)/(1-np.exp(-mu*Q_S/x)**2) ) )        
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


def calc_efficiency(row):
    T, W, Q_S, Q_M = row['Temperature'], row['Work'], row['System Heat'], row['Meter Heat']
    if W >= 0:
        if T < 1:
            return np.abs(W)/np.abs(Q_S)
        else:
            return (np.abs(W) + np.abs(Q_S)) /np.abs(Q_M) 
    else:
        return np.abs(Q_S)/np.abs(W)
if __name__ == "__main__":
    main()