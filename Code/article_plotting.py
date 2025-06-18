import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import scipy.constants as const
from plotting_data import multidata_preprocessing
import gc

data_dir = 'data/article_data_v3/'# 'data/article_data/used_data/'
#test_dir = 'data/testing_data/'

def main():
    #plot_efficiency(fname='images/test_images/efficiency_HE.pdf')
    #plot_info_efficiency()
    #plot_info_and_meter_level()
    #df = pd.read_csv(f'{data_dir}work_vs_time/params_vs_time_ratio=0.1_x=0.2.csv', skiprows=1)
    #plt.plot(df['Time'], df['Mutual Information'], label='Mutual Information')
    #plt.plot(df['Time'], -df['System Heat'], label='System Heat')
    #plt.xlim(0.06,0.1)
    #plt.show()
    #plot_efficiency_time()
    #df = pd.read_csv(f'{test_dir}params_vs_time_ratio=0.01_high_res_long.csv', skiprows=1)
    #df['Info Efficiency'] = (-df['System Heat']) / (df['Mutual Information'])
    #plt.plot(df['Time'], df['Info Efficiency'])
    #plt.show()
    #plot_work_vs_time()
    #plot_heatmap()
    #plot_probabilities()
    #plot_Jonas_data()
    heatmap_test()

def plot_work_vs_time():
    C_M = 1.5 # Energy splitting of the meter in units of kB*T_S
    # Load the data 
    df_list = [pd.read_csv(f'{data_dir}work_vs_time/params_vs_time_ratio={ratio}_x=0.2.csv', skiprows=1) for ratio in [0.01, 0.1, 1.0]]
    #df_list = [pd.read_csv(f"data/testing_data/params_vs_time_ratio={ratio}_high_res_long.csv", skiprows=1) for ratio in [0.01, 0.1, 1.0]]
    # Create a power column in each dataframe
    for df in df_list:
        df['Power'] = C_M*df['Work'] / df['Time']
        df['Power'] = df['Power']/df['Power'].abs().max() # Normalize the power to the maximum power
    label_list = [r'$g_\text{eff}^2/\Delta E$'+f' ={P}' for P in [0.01, 0.1, 1.0]]
    # Create a list of data to plot
    data_list = []
    for df in df_list:
        data_list.append([df['Time'], df['Power']])
    # Plot the data
    single_plot(
        data=data_list,
        fname='images/test_images/power_vs_time.pdf',
        xlabel=r' $\omega t_\text{m}/2\pi$',
        ylabel=r'$\Pi/|\Pi|_\text{max}$',#\quad [\Omega k_\text{B} T_\text{S}]$',
        legend_labels=label_list,
        save=True,
        show=False,
        legend=True,
        xlim=(0, 2),
        ylim=(None, None),
        type='line',
        point_size=1,
        figsize=(3.375, 2.5),
        linestyles=['-', '--', '-.'],
        hlines=[0.0],
        legend_loc='lower right'
        #legend_anchor=(1.025, 1.035)
    )

def plot_info_and_meter_level():
    # Load the data
    df = pd.read_csv(f'{data_dir}meter_level_vs_temp.csv')
    df2 = pd.read_csv(f'{data_dir}params_vs_temp.csv', skiprows=1)
    df2['info_efficiency'] = (-df2['System Heat'])/(df2['Mutual Information'])

    fig, ax1 = plt.subplots(figsize=(3.375, 2.5)) # Create a figure with a specific size

    ax1_color = 'black'
    ax1.plot(df2['Temperature'][1:], df2['info_efficiency'][1:], color=ax1_color, label=r'$\eta_\text{info}$')
    ax1.tick_params(axis='y', labelcolor=ax1_color)
    ax1.set_xlabel(r'Relative Temperature $T_\text{M}/T_\text{S}$')
    ax1.set_ylabel(r'$\eta_\text{info}$', color=ax1_color)
    ax1.set_ylim(0, 1)


    # Add vertical lines at the meter level crossings
    vlines = []
    for i in range(1, 9):
        temp = df['Temperature'].iloc[np.where(df['Meter Level'] == i)[0]]
        if len(temp) > 0:
            vlines.append(temp.iloc[0])
    for vline in vlines:
        ax1.axvline(x=vline, color='gray', linestyle='--', linewidth=0.5)

    # Create a  second y-axis for the meter levels
    ax2 = ax1.twinx()
    # Plot the meter levels on the right y-axis
    ax2_color = 'cornflowerblue'
    ax2.plot(df['Temperature'], df['Meter Level'], color=ax2_color, linestyle='-', label=r'$n^\prime$')
    ax2.set_ylabel(r'Meter State $n^\prime$', color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 8.1)
    ax2.spines['right'].set_color(ax2_color)  # Set the color of the right spine to match the y-axis color

    #fig.legend() # Add a legend to the figure
    plt.tight_layout() # Adjust the layout to prevent overlap

    # Save the figure
    plt.savefig('images/test_images/info_and_meter_level_option2.pdf', bbox_inches='tight', dpi=600)

    # Show the figure
    #plt.show()

def plot_efficiency_test():
    df_generator = (pd.read_csv(f'data/article_data/efficiency_data/efficiency_test_{C_M}.csv', skiprows=1) for C_M in [0.05, 0.1, 0.2, 0.3])
    data_list = []
    for df in df_generator:
        df = df[df['Work'] > 0].copy()
        df['efficiency'] = (1 + df['Meter Heat'] / df['System Heat'])
        data_list.append([df['Temperature'].to_numpy(), df['efficiency'].to_numpy()])
    data_list = []
    for df in df_list:
        df = df[df['Work'] > 0]
        df['efficiency'] = (1 + df['Meter Heat'] / df['System Heat'])
        data_list.append([df['Temperature'].to_numpy(), df['efficiency'].to_numpy()])
    # Find which of the temperature arrays contain the maximum value
    max_temp = 0
    for data in data_list:
        temp_var = np.max(data[0])
        if temp_var > max_temp:
            max_temp = temp_var
    
    # Create a Carnot and Curzon-Ahlborn efficiency array
    temp_list = np.linspace(0, max_temp, 2000)
    carnot = 1 - temp_list # Note that temp is stored as T_M/T_S where T_M < T_S
    curzon_ahlborn = 1 - np.sqrt(temp_list)
    # Append the carnot and curzon ahlborn efficiency arrays to the data list
    data_list.append([temp_list, carnot])
    data_list.append([temp_list, curzon_ahlborn])
    # Create the legend labels
    legend_labels = [r'$\hbar\omega = 0.05T_\text{S}$', r'$\hbar\omega = 0.1T_\text{S}$', r'$\hbar\omega = 0.2T_\text{S}$', r'$\hbar\omega = 0.3T_\text{S}$',\
                     r'$\eta_\text{C}$', r'$\eta_\text{CA}$']
    # Plot the data
    single_plot(
        data=data_list,
        fname='images/test_images/efficiency_test.pdf',
        xlabel=r'$T_\text{M}/T_\text{S}$',
        ylabel=r'$\eta_\text{info}$',
        legend_labels=legend_labels,
        save=True,
        show=False,
        legend=True,
        xlim=(0, None),
        ylim=(-0.0, 1.1)
    )


def plot_info_efficiency():
    df = pd.read_csv(f'{data_dir}params_vs_temp.csv', skiprows=1)
    T_S = 1
    df['info_efficiency'] = (-df['System Heat'])/(T_S*df['Mutual Information'])
    #df['info_efficiency'] = 1/(T_S*df['Mutual Information'])
    df['efficiency'] = (1 + df['Meter Heat'] / df['System Heat'])
    df['cop'] = (df['System Heat'] / df['Work']).abs()
    dataframe_entries = [['Temperature', 'info_efficiency'],
                         #['Temperature', 'efficiency'],
                         #['Temperature', 'cop']
                        ]
    legend_labels = [r'$\frac{W_\text{ext}}{T_\text{S} I}$',
                     #r'$\frac{W_\text{ext}+W_\text{meas}}{W_\text{ext}}$',
                     #r'$\left|\frac{W_\text{ext}}{W_\text{net}}\right|$'
                    ]
    # Find the temperature when the efficiency crosses zero but exclude the first value
    #zero_crossing = df['Temperature'].iloc[np.where(np.diff(np.sign(df['efficiency'])))[0]].to_numpy()

    single_plot(
        data = df,
        fname='images/test_images/info_efficiency_vs_temp.pdf',
        xlabel=r'Relative Temperature $T_\text{M}/T_\text{S}$',
        ylabel=r'$\eta_\text{info}$',
        dataframe_entries=dataframe_entries,
        save=True,
        show=False,
        legend_labels=legend_labels,
        ylim=(None, None),
        xlim=(0, 1),
        type='line',
        point_size=1,
        figsize=(3.375, 2.5),
        legend=False
    )  # , vlines=zero_crossing

def plot_probabilities():
    df = pd.read_csv(f'{data_dir}conditional_probabilities_meter_level.csv')
    data_list = [
        [df['Meter Level'], df['a']],
        [df['Meter Level'], df['b']],
        [df['Meter Level'], df['p0']],
        [df['Meter Level'], df['p1']]
    ]
    legend_labels = [
        r'$P(0|n,\omega t_{\text{m}}=0)$',
        r'$P(1|n,\omega t_{\text{m}}=0)$',
        r'$P(0|n,\omega t_{\text{m}}=\frac{\pi}{2})$',
        r'$P(1|n,\omega t_{\text{m}}=\frac{\pi}{2})$'
    ]
    line_colors = ['blue', 'orange', 'blue', 'orange' ]
    line_styles = ['scatter', 'scatter', 'line', 'line']
    # Reverse the order of the line styles
    line_styles = line_styles[::-1]
    # Find where the p0 and p1 cross each other
    crossing_points = np.where(np.diff(np.sign(df['p0'] - df['p1'])))[0]+1
    # Add vertical lines at the crossing points
    vlines = df['Meter Level'].iloc[crossing_points].to_numpy()
    # Add a text box entry for the crossing points
    text_box_entries = [r'$n^\prime$' for n in vlines]
    single_plot(data_list, fname='images/test_images/cond_prob_vs_meter_level.pdf', xlabel=r'Meter State $n$',\
                 ylabel='Conditional Probability', legend_labels=legend_labels, save=True, show=False, line_colors=line_colors,\
                    type=line_styles, vlines=vlines, xlim=(0, 30), text_box_entry=text_box_entries,\
                        legend_loc='right')


def plot_Jonas_data():
    col_names = ['eff', 'power', 'temp', 'dE', 'hw', 'geff', 't']
    df = pd.read_csv('data/jonas_collab_data/Pareto_HE_ergo_N15_eta.txt', header=None, names=col_names)
    #df = pd.read_csv('/home/hagge/Documents/Studier/ExJobb/Code/data/jonas_collab_data/Pareto_HE_ergo_N15_eta.txt', header=None, names=col_names)

    # Set some constants
    kB = 1
    hbar = 1
    T_S = 1

    # Convert the data. The parameters are stored as log10 values, so we need to convert them to normal values
    for param in col_names[2:]:
        df[param] = 10**df[param] # Fixes the log10 values
    # Convert the units
    df['temp'] = df['temp'] * T_S
    df['dE'] = df['dE'] * kB * T_S
    df['hw'] = df['hw'] * kB * T_S
    df['geff'] = df['geff'] * np.sqrt(kB * T_S)
    df['t'] = 2*np.pi*df['t']#*hbar/df['hw']
    # Choose what to plot
    dataframe_entries = [['eff', 'power'], ['eff', 'temp'], ['eff', 'dE'], ['eff', 'hw'], ['eff', 'geff'], ['eff', 't']]
    xlabel = r'$\eta_\text{info}$' # Everything is plotted against the efficiency
    # Set the ylabels
    ylabels = [r'$\Pi$', r'$T_\text{M}$', r'$\Delta E$', r'$h\omega$', r'$g_{\text{eff}}$', r'$t$']
    # Set the ylims to twice the median value for the dE and the t, no lims for the rest
    ylims = [(None, None), (None, None), (0.98*df['dE'].min(), 1.5*df['dE'].median()), (None, None), (None, None), (0.98*df['t'].min(), 1.5*df['t'].median())]


    #plot_subplots(df, fname='images/test_images/jonas_data_INFO.pdf', ncols=3, nrows=2, sharex=True, sharey=False,\
    #              xlabel=xlabel, ylabel=ylabels, save=True, dataframe_entries=dataframe_entries, show=False, ylim=ylims, type='scatter')
    #df['renormalized_power'] = df['power'] / df['geff']**2
    #single_plot(df, fname='images/test_images/jonas_data_renormalized_power.pdf', xlabel=r'$\eta_\text{info}$', ylabel=r'$\Pi/g_{\text{eff}}^2$',\
    #            dataframe_entries=[['eff', 'renormalized_power']], save=True, show=False, type='scatter', point_size=1)
    df['gt'] = df['geff']**2 * df['t']
    #single_plot(df, fname='images/test_images/gefft_vs_eff_info.pdf', xlabel=r'$\eta_\text{info}$', ylabel=r'$g_{\text{eff}}^2 t$',\
    #            dataframe_entries=[['eff', 'gt']], save=True, show=False, type='scatter', point_size=1)
    eff = df['eff'].to_numpy()
    power = df['power'].to_numpy()
    # Create a scatter plot of the power against the efficiency
    fig, ax = plt.subplots(figsize=(3.375, 2.5))
    # Color the points based on the value of gt
    c = ax.scatter(eff, power, c=df['dE'], cmap='viridis', s=1)
    # Add a colorbar
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label(r'$g_{\text{eff}}^2 t$')
    # Set the labels
    ax.set_xlabel(r'$\eta_\text{info}$')
    ax.set_ylabel(r'$\Pi$')
    plt.show()
    



def plot_efficiency(fname='images/test_images/efficiency_HE.pdf'):
    df_erg = pd.read_csv(f'{data_dir}/params_vs_temp.csv', skiprows=1)
    #df_erg=pd.read_csv('params_vs_temp_opt_eq_temp_long_ergotropy_V3.csv', skiprows=1)
    # Drop the first row
    df_erg = df_erg.drop(df_erg.index[0])
    # Find the heat engine efficiency
    data = df_erg[df_erg['Work'] > 0]
    heat_engine_eff = (1 + data['Meter Heat'] / data['System Heat']).to_numpy()
    heat_engine_temps = data['Temperature'].to_numpy()

    ## Find the heat valve efficiency
    data = df_erg[(df_erg['Work'] < 0) & (df_erg['Temperature'] < 1)]
    heat_valve_eff = (np.abs(data['System Heat'] / data['Work'])).to_numpy()
    heat_valve_temps = data['Temperature'].to_numpy()

    ## Find the refrigerator efficiency
    data = df_erg[(df_erg['Work'] < 0) & (df_erg['Temperature'] > 1)]
    heat_refrigerator_eff = (np.abs(data['System Heat'] / data['Work'])).to_numpy()
    heat_refrigerator_temps = data['Temperature'].to_numpy()

    # Find the Carnot COP of cooling in the heat valve regime

    
    # Find the Carnot and Curzon-Ahlborn efficiencies
    carnot = 1 - heat_engine_temps
    curzon_ahlborn = 1 - np.sqrt(heat_engine_temps)
    data_list = [
        [heat_engine_temps, carnot],
        [heat_engine_temps, curzon_ahlborn],
        [heat_engine_temps, heat_engine_eff]
    ]
    print(f"Carnot at last temperature: {carnot[-1]}, Curzon-Ahlborn at last temperature: {curzon_ahlborn[-1]}")
    legend_labels = [
        #r'COP',
        r'$\eta_\text{C}$',
        r'$\eta_\text{CA}$',
        r'$\eta_\text{HE}$'
    ]
    linestyles = ['--', ':', '-']
    xlim = (0, heat_engine_temps.max())
    ylim = (-0.0, 1.1)
    single_plot(data_list, fname=f'{fname}', xlabel=r'Relative Temperature $T_\text{M}/T_\text{S}$', ylabel=r'Efficiency', \
             legend=True, xlim=xlim, ylim=ylim, save=True, legend_labels=legend_labels,\
             figsize=(3.375, 2.5), show=False, vlines=[heat_engine_temps[-1]], type='line',\
                  linestyles=linestyles, legend_loc='lower left')

def plot_efficiency_time(fname='images/test_images/efficiency_time.pdf'):
    T_S = 1.0 # Set the temperature of the system
    # Load the data
    #df_list = [pd.read_csv(f'{data_dir}efficiency_over_time/params_vs_time_ratio={ratio}_high_res.csv', skiprows=1) for ratio in [0.01, 0.1, 1.0]]
    #df_list = [pd.read_csv(f'{test_dir}params_vs_time_ratio={ratio}_high_res_long.csv', skiprows=1) for ratio in [0.01, 0.1, 1.0]]
    df_list = [pd.read_csv(f'{data_dir}work_vs_time/params_vs_time_ratio={ratio}_x=0.2.csv', skiprows=1) for ratio in [0.01, 0.1]]#, 1.0]]

    # Curzon-Ahlborn efficiency
    eta_CA = 1 - np.sqrt(0.1) # The temp ratio is fixed at 0.1, so we can calculate the Curzon-Ahlborn efficiency directly
    # At zero time the values of System Heat, Meter Heat and Mutual Information are all zero, 
    # so we make them non-zero by adding a small value to them

    # First select only the rows where the Work is positive, i.e. the heat engine regime, into a new list
    for df in df_list:
        df.drop(df[df['Work'] < 0].index, inplace=True)

    #eps = 1e-20
    #for df in df_list:
    #    df['System Heat'] = df['System Heat'] + eps
    #    df['Meter Heat'] = df['Meter Heat'] + eps
    #    df['Mutual Information'] = df['Mutual Information'] + eps

    # Find the heat engine efficiency
    # Create a new column for the efficiency
    for df in df_list:
        df['Efficiency'] = (1 + df['Meter Heat'] / df['System Heat'])/ eta_CA # Normalize the efficiency to the Curzon-Ahlborn efficiency
    # Create a list of data to plot
    #data_list = [[df['Time'], df['Efficiency']]]
    data_list = [[df_list[0]['Time'], df_list[0]['Efficiency']],
                  [df_list[1]['Time'], df_list[1]['Efficiency']]]
    # Create a list of legend labels
    legend_labels = [r'$g_\text{eff}^2/\Delta E=0.01$', r'$g_\text{eff}^2/\Delta E=0.1$']
    # Plot the data
    single_plot(
        data=data_list,
        fname=fname,
        xlabel=r'$\omega t_\text{m}/2\pi$',
        ylabel=r'$\eta_\text{HE}/\eta_\text{CA}$',
        legend_labels=legend_labels,
        save=True,
        show=False,
        legend=True,
        xlim=(0, 0.5),
        ylim=(0, None),
        type='line',
        point_size=1,
        figsize=(3.375, 2.5),
        linestyles=['-', '--'],
        legend_loc='lower right'
    )
    # Now also plot the information efficiency
    C_S = 4.33
    gtm = False
    # This is stupid but reload the dataframes to avoid having to change the code above
    #df_list = [pd.read_csv(f'{test_dir}params_vs_time_ratio={ratio}_high_res_long.csv', skiprows=1) for ratio in [0.01, 0.1, 1.0]]
    df_list = [pd.read_csv(f'{data_dir}work_vs_time/params_vs_time_ratio={ratio}_x=0.2.csv', skiprows=1) for ratio in [0.01, 0.1]]#, 1.0]]
    for df in df_list:
        df['Info Efficiency'] = (-df['System Heat']) / (T_S*df['Mutual Information'])
        # Find where the info efficiency is exactly zero and set it to the smallest non-zero value
        df['Info Efficiency'] = df['Info Efficiency'].replace(0, np.min(df['Info Efficiency'][df['Info Efficiency'] > 0]))
    # Create a list of data to plot
    #data_list = [[df['Time'], df['Info Efficiency']]]
    y_vals = [0.01, 0.1]#, 1.0]
    if gtm:
        for df,y in zip(df_list, y_vals):
            df['Time'] = df['Time']*2*np.pi*y*C_S
    data_list = [[df_list[0]['Time'], df_list[0]['Info Efficiency']],
                  [df_list[1]['Time'], df_list[1]['Info Efficiency']],
                  #[df_list[2]['Time'], df_list[2]['Info Efficiency']]
                ]
    # Create a list of legend labels
    legend_labels = [r'$g_\text{eff}^2/\Delta E=0.01$',
                     r'$g_\text{eff}^2/\Delta E=0.1$',
                     #r'$g_\text{eff}^2/\Delta E=1.0$'
                     ]
    # Create a filename for the plot
    word_list = fname.split('/')
    word_list[-1] = 'info_' + word_list[-1]
    fname = '/'.join(word_list)
    if gtm:
        xlabel = r'$g_{\text{eff}}^2 t_\text{m}/\hbar$'
    else:
        xlabel = r'$\omega t_\text{m}/2\pi$'
    single_plot(data=data_list, 
                fname=fname,
                #xlabel=r'$\omega t_\text{m}/2\pi$',
                xlabel=xlabel,
                ylabel=r'$\eta_\text{info}$',
                #legend_labels=[r'$\eta_\text{info}$'],
                save=True,
                show=False,
                legend_labels=legend_labels,
                legend=True,
                xlim=(0, 0.5),
                ylim=(0, 1.05),
                type='line',
                point_size=1,
                figsize=(3.375, 2.5),
                linestyles=['-','--'],
                legend_loc='lower right'
    )


def plot_heatmap():
    # Data directory for the heatmap data
    # Read the data for the heatmap
    tau_vals = np.array([1e-06, 0.125, 0.25, 0.5])
    df_list = [pd.read_csv(f'{data_dir}multidata_test_tau={tau}_R=0_net.csv', index_col=0) for tau in tau_vals]
    #df_list = [pd.read_csv(f'data/article_data/Pirkkalainen/multidata_coupling_Pirkkalainen_tau={tau}_net.csv', index_col=0) for tau in tau_vals]
    # Create a meshgrid list for the heatmaps
    meshgrid_list = []
    xlim = 0.5
    ylims = [50, 5, 5, 5]
    #ylims = [1]*4
    #C_M = 0.072
    C_M = 1.50

    for i, df in enumerate(df_list):
        df = df[df.index.astype(float) <= ylims[i]] # Remove the rows that are above the ylims 
        index_values = df.index.astype(float) 
        column_values = df.columns.astype(float)
        ind_xlim = next((i for i, val in enumerate(column_values) if val > xlim), len(column_values))
        df = df.iloc[:, :ind_xlim]
        column_values = column_values[:ind_xlim]
        X, Y = np.meshgrid(column_values, index_values)
        # Simulation done with hbar=kb=T_S=1, and we want the power in units of 2*pi*hbar/(kB*T_S)^2 
        # The data is actually work, so also need to multiply by the factor C_M/tau to get the power
        Z = C_M*df.to_numpy()/(tau_vals[i]) # Convert the data to the correct units
        if i == 0:
            Z = Z*1000 # Manually upscale the first heatmap to make it more visible
        meshgrid_list.append([X, Y, Z])


    text_box_entries = [r'$ \omega t_{\text{m}} = 2\pi\cdot 10^{-6} $', r'$ \omega t_{\text{m}} = \pi / 4$',\
                        r'$ \omega t_{\text{m}} = \pi / 2$', r'$ \omega t_{\text{m}} = \pi$']
    plot_subplots(meshgrid_list, fname='images/test_images/heatmap_test.pdf', ncols=2, nrows=2, sharex=True, sharey=True,\
                  xlabel=r'Relative Temperature $T_{\text{M}}/T_{\text{S}}$', ylabel=r'$g_{\text{eff}}^2/\Delta E$', title=None, legend=False,\
                    figsize=(3.375, 2.5), show=False, type='heatmap', ylim=ylims, cbar_label=r'Power $\Pi/\Omega k_\text{B}T_\text{S}$',\
                          panel_labels=['(a)', '(b)', '(c)', '(d)'], text_box_entries=text_box_entries)



def heatmap_test2():
    import seaborn as sns
    df = pd.read_csv("data/article_data_v3/multidata_coupling_tau=1e-06_v3.csv")

    # Pivot for heatmap
    heatmap_data = df.pivot(index="P2_over_QS", columns="T", values="W")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={"label": "Net Work (W)"})
    plt.xlabel(r"Temperature Ratio $T_\text{M}/T_\text{S}$")
    plt.ylabel(r"$g_\text{eff}^2 / \Delta E$")
    plt.title("Net Work vs Temperature and Coupling")
    plt.tight_layout()
    plt.show()

def heatmap_test():
    import cmocean
    from matplotlib.colors import TwoSlopeNorm
    df = pd.read_csv("data/article_data_v3/multidata_coupling_tau=0.125_v3.csv")
    df = df[df['P2_over_QS'] <= 5]  # Limit the P2_over_QS to 50 for better visualization
    df = df[df['T'] <= 0.5]  # Limit the T to 5 for better visualization
    value = "W"
    pivot = df.pivot(index='P2_over_QS', columns='T', values=value)
    fig, ax = plt.subplots(figsize=(8, 6))

    #cmap = 'viridis'  # Choose a colormap
    vmin = np.nanmin(pivot.values)
    vmax = np.nanmax(pivot.values)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # Normalize the colormap to center around zero
    cmap = cmocean.cm.balance  # Use a colormap that is suitable for diverging data
    c = ax.imshow(
        pivot.values,
        aspect='auto',
        origin='lower',  # Keep origin upper for correct flip
        rasterized=True,
        cmap=cmap,
        norm=norm,
        extent=[
            pivot.columns.min(), pivot.columns.max(),
            pivot.index.min(), pivot.index.max()
        ]
    )

    # Flip y-axis
    #ax.invert_yaxis()

    # Set axis labels
    ax.set_xlabel('Temperature Ratio (x)')
    ax.set_ylabel(r'$P^2 / \Delta E$')

    # Format ticks (approx. 5 per axis, rounded to 2 sig. figs)
    def format_ticks(values):
        step = max(1, len(values) // 4)
        return np.round(values[::step], 2)

    xticks = format_ticks(pivot.columns.values)
    yticks = format_ticks(pivot.index.values)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Ensure clean labels
    ax.set_xticklabels([f"{x:.2g}" for x in xticks])
    ax.set_yticklabels([f"{y:.2g}" for y in yticks])

    # Add colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(value)

    #if title:
    #    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def single_plot(data, fname='single_plot.png', xlabel='x', ylabel='y', title=None, legend=True, xlim=None, ylim=None, grid=False,\
             save=True, legend_labels=None, dataframe_entries=None, figsize=(3.375, 2.5), show=False, vlines=None, hlines=None,\
                text_box_entry=None, type='line', point_size=None, line_colors=None, linestyles=None,\
                legend_loc='upper right', legend_anchor=None):
    """Creates a single plot with the given data and saves it to a file. If any of the data
    are dataframes, you must specify the dataframe_entires parameter to specify which columns to plot.

    Args:
        data (list): List of data to plot. Each element can be a list, numpy array, or pandas dataframe, containing both x and y data.
        save (bool, optional): Whether to save the plot to a file. Defaults to True.
        fname (str, optional): Filename of the saved file. Defaults to 'single_plot.png'.
        xlabel (str, optional): Label of x-axis. Defaults to 'x'.
        ylabel (str, optional): Label of y-axis. Defaults to 'y'.
        title (_type_, optional): Title. Defaults to None.
        legend (bool, optional): Legend on/off. Defaults to True.
        xlim (_type_, optional): Limit x-axis. Defaults to None.
        ylim (_type_, optional): Limit y-axis. Defaults to None.
        grid (bool, optional): Grid off/one. Defaults to False.
        legend_labels (_type_, optional): Legend entries, must be sorted in the same order as data. Defaults to None.
        dataframe_entries (_type_, optional): Dataframe entries to plot. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (3.375, 2.5).
        show (bool, optional): Show the plot. Defaults to False.
        vlines (list, optional): Vertical lines to plot. Defaults to None.
        hlines (list, optional): Horizontal lines to plot. Defaults to None.
        text_box_entry (list, optional): Text box entry. Defaults to None.
        type (str, list, optional): Type of plot. Defaults to 'line' for a line plot. Also supports 'scatter' for scatter plots.
        point_size (int, optional): Point size for scatter plot. Defaults to None.
        line_colors (list, optional): List of colors for the lines. Defaults to None.
        linestyles (list, optional): List of linestyles for the lines. Defaults to None.
        legend_loc (str, optional): Location of the legend. Defaults to 'upper right'.
        legend_anchor (tuple, optional): Anchor point for the legend. Defaults to None.
    """

    fig, ax = plt.subplots(figsize=figsize)
    #ax.axvspan(0, vlines[0], facecolor='gray', alpha=0.5, edgecolor=None, zorder=1 ) if vlines else None
    #ax.text(3, 0.0, 'Passive')
    #ax.text(11, 0.0, 'Active')
    # First check if the data contains dataframes and if so, if the dataframe_entries are specified
    if hlines:
        for hline in hlines:
            ax.axhline(y=hline, color='k', linestyle='-', linewidth=1.0)
    if vlines:
        for vline in vlines:
            ax.axvline(x=vline, color='k', linestyle='--', linewidth=0.5, zorder=1)
    if not isinstance(line_colors, list):
        line_colors = [line_colors] * len(data)
    if not isinstance(type, list):
        type = [type] * len(data)
    if not isinstance(xlabel, list):
        xlabel = [xlabel] * len(data)
    if not isinstance(ylabel, list):
        ylabel = [ylabel] * len(data)
    if isinstance(data, pd.DataFrame):
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * len(dataframe_entries)
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * len(dataframe_entries)
        if dataframe_entries is None:
            raise ValueError("dataframe_entries must be specified if data is a dataframe.")
        for i, entry in enumerate(dataframe_entries):
            x = data[entry[0]]
            y = data[entry[1]]
            if type[i] == 'line':
                ax.plot(x, y, label=legend_labels[i] if legend_labels else None, color=line_colors[i], zorder=2, linestyle=linestyles[i] if linestyles else None)
            elif type[i] == 'scatter':
                ax.scatter(x, y, label=legend_labels[i] if legend_labels else None, color=line_colors[i], zorder=2)
            ax.set_xlabel(xlabel[i])
            ax.set_ylabel(ylabel[i])
    elif any(isinstance(d, pd.DataFrame) for d in data):
        if dataframe_entries is None:
            raise ValueError("dataframe_entries must be specified if any of the data are dataframes.")
        for i, d in enumerate(data):
            if isinstance(d, pd.DataFrame):
                x = d[dataframe_entries[i][0]]
                y = d[dataframe_entries[i][1]]
            else:
                x = d[0]
                y = d[1]
            if type[i] == 'line':
                ax.plot(x, y, label=legend_labels[i] if legend_labels else None, linestyle=linestyles[i] if linestyles else None)
            elif type[i] == 'scatter':
                if point_size is not None:
                    ax.scatter(x, y, label=legend_labels[i] if legend_labels else None, s=point_size, zorder=2)
                else:
                    ax.scatter(x, y, label=legend_labels[i] if legend_labels else None, zorder=2)
            ax.set_xlabel(xlabel[i])
            ax.set_ylabel(ylabel[i])
    else:
        for i, d in enumerate(data):
            x = d[0]
            y = d[1]
            if type[i] == 'line':
                ax.plot(x, y, label=legend_labels[i] if legend_labels else None, linestyle=linestyles[i] if linestyles else None)
            elif type[i] == 'scatter':
                if point_size is not None:
                    ax.scatter(x, y, label=legend_labels[i] if legend_labels else None, s=point_size, zorder=2)
                else:
                    ax.scatter(x, y, label=legend_labels[i] if legend_labels else None, zorder=2)
            ax.set_xlabel(xlabel[i])
            ax.set_ylabel(ylabel[i])
            #ax.plot(x, y, label=legend_labels[i] if legend_labels else None)
    if text_box_entry:
        for i, entry in enumerate(text_box_entry):
            ax.text(0.36, 0.95 - i*0.05, entry, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right', color='black',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none', pad=0))
    if legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor) if legend_anchor else ax.legend(loc=legend_loc)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid:
        ax.grid()
    if title:
        ax.set_title(title)
    if save:
        plt.savefig(fname, bbox_inches='tight')
    if show:
        plt.show()

    
    return fig, ax


def plot_subplots(data, fname='subplots.png', ncols=1, nrows=1, sharex=False, sharey=False, xlabel='x', ylabel='y', title=None,\
                    legend=True, xlim=None, ylim=None, grid=False, save=True, legend_labels=None, dataframe_entries=None,\
                    figsize=(3.375, 2.5), show=False, type='line', vlines=None, hlines=None, panel_labels=None, cbar_label=None,\
                    text_box_entries=None):
    """Creates a single plot with the given data and saves it to a file. If any of the data
    are dataframes, you must specify the dataframe_entires parameter to specify which columns to plot.

    Args:
        data (list): List of data to plot. Each element can be a list, numpy array, or pandas dataframe, containing both x and y data.
        fname (str, optional): Filename of the saved file. Defaults to 'single_plot.png'.
        ncols (int, optional): Number of columns. Defaults to 1.
        nrows (int, optional): Number of rows. Defaults to 1.
        sharex (bool, optional): Share x-axis. Defaults to False.
        sharey (bool, optional): Share y-axis. Defaults to False.
        type (str, optional): Type of plot. Defaults to 'line' for a line plot. Also supports 'scatter' for scatter plots, and 'heatmap' for heatmaps.
        vlines (list, optional): Vertical lines to plot. Defaults to None.
        hlines (list, optional): Horizontal lines to plot. Defaults to None.
        xlabel (str, optional): Label of x-axis. Defaults to 'x'.
        ylabel (str, optional): Label of y-axis. Defaults to 'y'.
        title (_type_, optional): Title. Defaults to None.
        legend (bool, optional): Legend on/off. Defaults to True.
        xlim (_type_, optional): Limit x-axis. Defaults to None.
        ylim (_type_, optional): Limit y-axis. Defaults to None.
        grid (bool, optional): Grid off/one. Defaults to False.
        save (bool, optional): Whether to save the plot to a file. Defaults to True.
        legend_labels (_type_, optional): Legend entries, must be sorted in the same order as data. Defaults to None.
        dataframe_entries (_type_, optional): Dataframe entries to plot. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (3.375, 2.5).
        show (bool, optional): Show the plot. Defaults to False.
        panel_labels (list, optional): Panel labels. Defaults to None.
        cbar_label (str, optional): Colorbar label. Defaults to None.
        text_box_entries (list, optional): Text box entries. Defaults to None.
    """
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    # Check if it is only a single dataframe, in which case plot all dataframe_entries in individual subplots
    if isinstance(data, pd.DataFrame):
        if not isinstance(xlabel, list):
            xlabel = [xlabel] * len(dataframe_entries)
        if not isinstance(ylabel, list):
            ylabel = [ylabel] * len(dataframe_entries)
        if not isinstance(type, list):
            type = [type] * len(dataframe_entries)
        if dataframe_entries is None:
            raise ValueError("dataframe_entries must be specified if data is a dataframe.")
        for i, entry in enumerate(dataframe_entries):
            ax = axs[i//ncols, i%ncols] if ncols > 1 else axs[i]
            x = data[entry[0]]
            y = data[entry[1]]
            if type[i] == 'line':
                ax.plot(x, y, label=legend_labels[i] if legend_labels else None)
            elif type[i] == 'scatter':
                ax.scatter(x, y, label=legend_labels[i] if legend_labels else None)
            elif type[i] == 'heatmap':
                c = ax.imshow(y, aspect='auto')
                fig.colorbar(c, ax=ax)
            ax.set_xlabel(xlabel[i])
            ax.set_ylabel(ylabel[i])
            ax.set_ylim(ylim[i][0], ylim[i][1]) if ylim is not None else None
            ax.set_xlim(xlim[i][0], xlim[i][1]) if xlim is not None else None

    # Check if any of the data are dataframes and if so, if the dataframe_entries are specified
    elif any(isinstance(d, pd.DataFrame) for d in data):
        if dataframe_entries is None:
            raise ValueError("dataframe_entries must be specified if any of the data are dataframes.")
        for i, d in enumerate(data):
            if isinstance(d, pd.DataFrame):
                x = d[dataframe_entries[i][0]]
                y = d[dataframe_entries[i][1]]
            else:
                x = d[0]
                y = d[1]
            if type == 'line':
                axs.plot(x, y, label=legend_labels[i] if legend_labels else None)
            elif type == 'scatter':
                axs.scatter(x, y, label=legend_labels[i] if legend_labels else None)
            elif type == 'heatmap':
                c = axs.imshow(y, aspect='auto')
                fig.colorbar(c, ax=axs)
    else:
        from matplotlib.ticker import ScalarFormatter

        axs = np.array(axs).reshape(nrows, ncols)


        # Add spacing on the right to make room for colorbars
        #fig.subplots_adjust(right=0.88, wspace=0.3, hspace=0.3)
        fig.subplots_adjust(
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.85,        # leave room for colorbar above if needed
            wspace=0.18,     # reduced horizontal space
            hspace=0.22      # reduced vertical space
        )

        for i, d in enumerate(data):
            row = i // ncols
            col = i % ncols
            ax = axs[row, col]
            X, Y, Z = d

            # Per-panel normalization centered at midpoint of data (not zero)
            zmin, zmax = Z.min(), Z.max()
            zcenter = 0 if (zmin < 0 < zmax) else (zmin + zmax) / 2
            norm = mcolors.TwoSlopeNorm(vmin=zmin, vcenter=zcenter, vmax=zmax)

            cmap = 'viridis'
            c = ax.pcolormesh(X, Y, Z, shading='auto', rasterized=True, cmap=cmap, norm=norm)

            # Axis labels and tick labels only on outer edges
            if row == 0:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)  # keep ticks, hide tick labels
            elif row == nrows - 1:
                ax.set_xlabel(xlabel)

            if xlim is not None:
                ax.set_xlim(X.min(), xlim[i])
            if ylim is not None:
                ax.set_ylim(0, ylim[i])
            if grid:
                ax.grid()

            if panel_labels:
                ax.text(+0.01, 1.1, panel_labels[i], transform=ax.transAxes,
                        verticalalignment='center', horizontalalignment='left', color='black',\
                              bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='none',pad=0))


        if text_box_entries is not None:
            vcoord = 1.1
            hcoord = 0.2
            # Iterate over the subplots and the text box entries
            for i, (ax, text) in enumerate(zip(axs.flat, text_box_entries)):
                # Place the text box in the top right corner of each subplot
                if i == 0:
                    ax.text(hcoord, vcoord, text, transform=ax.transAxes,
                            verticalalignment='center', horizontalalignment='left', color='black',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.0,
                                    edgecolor='none', pad=0.0))
                    ax.text(0.95, 0.95, r'($\times 10^{3}$)', transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='right', color='black',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                                    edgecolor='none', pad=0.2))
                    #ax.text(0.65, 1.15, text+'\n'+r'(power $\times 10^{3}$)', transform=ax.transAxes,
                    #        verticalalignment='top', horizontalalignment='right', color='black',
                    #        bbox=dict(boxstyle='round', facecolor='white', alpha=1.0,
                    #                edgecolor='none', pad=0.2))
                else:
                    ax.text(hcoord, vcoord, text, transform=ax.transAxes,
                            verticalalignment='center', horizontalalignment='left', color='black',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.0,
                                    edgecolor='none', pad=0.0))

        # If sharex is True, remove the individual x-axis labels and use a common x-axis label
        if sharex:
            for ax in axs.flat:
                ax.set_xlabel('')
            fig.supxlabel(xlabel, y=-0.03, size=size)  # Move x-label further down to avoid overlap
        # If sharey is True, remove the individual y-axis labels and use a common y-axis label
        if sharey:
            for ax in axs.flat:
                ax.set_ylabel('')
            fig.supylabel(ylabel, x=-0.01, size=size)  # Move y-label further left to avoid overlap
        cbar_ax = fig.add_axes([0.15, 0.93, 0.7, 0.02])  # push to top
        from matplotlib.ticker import FormatStrFormatter
        cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal', format=FormatStrFormatter('%.2f'))#format=ScalarFormatter(useMathText=True))
        vmin, vmax = d[2].min(), d[2].max()
        cbar.set_ticks([vmin, 0, vmax])
        cbar.update_ticks()
        cbar.set_label(cbar_label, labelpad=6)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')       # Move ticks to the top
        if save:
            plt.savefig(fname, bbox_inches='tight', dpi=600)
        if show:
            plt.show()
        return fig, axs

if __name__ == "__main__":
    #plt.rcParams.update({
    #    'font.size': 8,              # Base font size (10 pt for APS)
    #    'axes.titlesize': 10,         # Same as font size for axis titles
    #    'axes.labelsize': 8,         # Same as font size for axis labels
    #    'xtick.labelsize': 8,         # Slightly smaller tick labels for x-axis
    #    'ytick.labelsize': 8,         # Slightly smaller tick labels for y-axis
    #    'legend.fontsize': 8,         # Slightly smaller legend font size
    #    'figure.titlesize': 10,       # Consistent with main font for figure titles
    #    'font.family': 'serif',       # APS typically uses serif fonts
    #    'figure.dpi': 600,            # High-resolution for screen display
    #    'savefig.dpi': 600,           # High-resolution for saving figures (print quality)
    #    'lines.linewidth': 1,         # Moderate line width for plots
    #    'axes.linewidth': 1,          # Border line width for axes
    #    'xtick.major.width': 0.8,     # Major tick mark width for x-axis
    #    'ytick.major.width': 0.8,     # Major tick mark width for y-axis
    #    'xtick.direction': 'in',      # Ticks pointing inward on x-axis
    #    'ytick.direction': 'in',      # Ticks pointing inward on y-axis
    #    'font.serif': ['CMU Serif', 'Computer Modern Roman'], # Use Computer Modern font, with fallback
    #    'lines.markersize': 1         # Marker size for scatter plots and data points
    #})
    size = 9
    plt.rcParams.update({
        'font.size': size,             # Base font size (e.g., 8.5 pt for typical double-column APS)
        'axes.titlesize': size,        # Same as font size for axis titles
        'axes.labelsize': size,        # Same as font size for axis labels
        'xtick.labelsize': size,         # Slightly smaller tick labels for x-axis (e.g., 7 pt)
        'ytick.labelsize': size,         # Slightly smaller tick labels for y-axis (e.g., 7 pt)
        'legend.fontsize': size,         # Slightly smaller legend font size (e.g., 8 pt)
        'figure.titlesize': size,      # Consistent with main font for figure titles
        'font.family': 'serif',       # APS typically uses serif fonts
        'figure.dpi': 600,            # High-resolution for screen display
        'savefig.dpi': 600,           # High-resolution for saving figures (print quality)
        'lines.linewidth': 1,         # Moderate line width for plots
        'axes.linewidth': 1,          # Border line width for axes
        'xtick.major.width': 0.8,     # Major tick mark width for x-axis
        'ytick.major.width': 0.8,     # Major tick mark width for y-axis
        'xtick.direction': 'in',      # Ticks pointing inward on x-axis
        'ytick.direction': 'in',      # Ticks pointing inward on y-axis
        #'font.serif': ['CMU Serif', 'Computer Modern Roman'], # Use Computer Modern font, with fallback
        'text.usetex': True,              # Use LaTeX for text rendering
        "text.latex.preamble": r"\usepackage{amsmath}",
        'legend.frameon': False,  # No frame around the legend
        'legend.facecolor': 'none',  # Transparent legend background
        #'mathtext.fontset': 'cm',  # Use Computer Modern font for math text
        'lines.markersize': 1         # Marker size for scatter plots and data points
        
    })
    main()