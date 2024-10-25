import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import os

# Set the style of the plot
sns.set_style(style='white')
sns.set_context(context='paper', font_scale=2.0)
sns.set_palette(palette='colorblind')

symbol_dict = {'temperature': r'$T$', 'hw/de': r'$\hbar\omega/\Delta E$',\
                'time': r'$\tau$', 'coupling strength': r'$g$', "work": r'$W$',\
                    "system heat": r'$Q_{\text{sys}}$', "meter heat": r'$Q_{meter}$',\
                    "information": r'$I$', "observer information": r'$I_{\text{obs}}$',\
                    "mutual information": r'$I_{\text{mut}}$', "entropy": r'$S$'}

x_axes = ['Temperature', 'hw/dE', 'Time', 'Coupling Strength']
file_endings = ['temp', 'omega_per_delta_E', 'time', 'coupling']

for x_axis, file_ending in zip(x_axes, file_endings):
    # Check if the directory exists
    if not os.path.exists(f'images/param_vs_{file_ending}'):
        os.makedirs(f'images/param_vs_{file_ending}')
    # Import data from csv file
    sim_run = "_fixed"
    try:
        data = pd.read_csv(f'data/params_vs_{file_ending}{sim_run}.csv', skiprows=1)
    except FileNotFoundError:
        print(f"File {file_ending}{sim_run} not found")
        continue
    fixed_params = pd.read_csv(f'data/params_vs_{file_ending}{sim_run}.csv', nrows=1)
    title_string = ', '.join([str(x).strip() for x in fixed_params.columns])

    for param in data.columns[1:]:
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=x_axis, y=param, data=data)
        plt.title(title_string)
        # Set x and y labels with the correct symbols
        plt.xlabel(f'{symbol_dict[x_axis.lower()]}')
        plt.ylabel(f'{symbol_dict[param.lower()]}')
        sns.despine()
        plt.savefig(f'images/param_vs_{file_ending}/{param}{sim_run}.png')
        plt.close()
    # Select all columns that contain the word "information"
    info_cols = [col for col in data.columns if 'information' in col.lower()]
    # Plot Work / Information for all columns that contain the word "information"
    if info_cols:
        plt.figure(figsize=(12, 8))
        for i, info_col in enumerate(info_cols):
            data[f'Work/{info_col}'] = data['Work'] / data[info_col]
            plt.subplot(len(info_cols), 1, i + 1)
            sns.lineplot(x=x_axis, y=info_col, data=data)
            # Set x and y labels with the correct symbols
            plt.xlabel(f'{symbol_dict[x_axis.lower()]}')
            plt.ylabel(f'{symbol_dict["work"]} / {symbol_dict[info_col.lower()]}')
            plt.title(title_string)
            plt.xlabel(f'{x_axis}')
            sns.despine()
        plt.tight_layout()
        plt.savefig(f'images/param_vs_{file_ending}/Q_{info_col}{sim_run}.png')
        plt.close()