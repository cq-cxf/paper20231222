import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_excel(r'./Result of Geographical origin classification accuracy.xlsx')  # Update the path to your file

# Process the data
data = data.set_index(data.columns[0])  # Set the first column as the index
data = data.apply(pd.to_numeric, errors='coerce')  # Convert data to numeric, coercing errors

# Define accuracy categories
def categorize_accuracy(acc):
    if acc >= 0.95:
        return '>= 95%'
    elif acc >= 0.85:
        return '>= 85% and < 95%'
    else:
        return '< 85%'

# Apply categorization to each column
categorized_data = data.applymap(categorize_accuracy)

# Count the occurrences of each category for each model
category_counts = {col: categorized_data[col].value_counts() for col in categorized_data}

# Create a DataFrame from the counts
counts_df = pd.DataFrame(category_counts).fillna(0).reindex(['>= 95%', '>= 85% and < 95%', '< 85%'])

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(15, 15), dpi=600, facecolor='none')
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Define distinct colors for each category
colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
          '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#bcf60c', '#e6194B', '#46f0f0', '#ffe119', '#f032e6', '#808080', '#f58231', '#9a6324', '#fabebe']

# Iterating over each subplot
for i, ax in enumerate(axs.flatten()):
    model = data.columns[i]
    ax.bar(counts_df.index, counts_df[model], color=colors[i*3:(i+1)*3], alpha=0.7)
    ax.set_title(model, fontsize=12, fontname='Times New Roman')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=12)

# Making the figure background transparent
fig.patch.set_alpha(0.0)

# Save the figure with a transparent background
plt.savefig(r'./Figure 3. Classification_accuracy_plots.png', bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.show()
