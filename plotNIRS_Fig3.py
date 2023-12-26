import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv(r'./NIRS_dataset.csv')  # Make sure to replace with the actual file path

# Extract the wavelengths; these are typically the column names in the data
wavelengths = data.columns[2:].astype(int)
origins = ['All Origins'] + list(data['OP'].unique())

# Set the color map for plotting
color_map = plt.cm.viridis

# Create the subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=600, facecolor='none')
titles = ["All Origins", "Wenshan", "Honghe", "Kunming", "Qujing", "Yuxi"]

# Plot each subplot
for i, ax in enumerate(axs.flatten()):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Wavelength(cm$^{-1}$)', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('Absorbance', fontsize=12, fontname='Times New Roman')
    ax.grid(True)
    ax.title.set_text(titles[i])
    ax.title.set_fontsize(12)
    ax.title.set_fontname('Times New Roman')

    if titles[i] == "All Origins":
        for j in range(len(data)):
            ax.plot(wavelengths, data.iloc[j, 2:], linewidth=0.5)
    else:
        filtered_data = data[data['OP'] == titles[i].replace(" ", "")]
        for j in range(len(filtered_data)):
            ax.plot(wavelengths, filtered_data.iloc[j, 2:], linewidth=0.5)

# Adjust layout to make room for the figure title and save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Figure 2. NIRS_spectra.png', format='png', bbox_inches='tight', transparent=True)
plt.show()
