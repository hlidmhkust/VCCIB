import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

plt.rcParams['figure.figsize'] = (50, 20)
plt.rcParams['font.size'] = 25
# Generate some random data

data = [[0.9215, 0.9150, 0.9342, 0.9424, 0.9468,0.9478],[0.9433, 0.9400, 0.9516,0.9503, 0.9555,0.9436],
        [0.9588, 0.9566, 0.9571, 0.9503, 0.9512, 0.9593],[0.9637,0.9582, 0.9571,0.9506,0.9482,0.9626],[0.9648, 0.9582, 0.9558, 0.9482, 0.9441, 0.9617]]
data2 = [[0.9591,0.9608,0.9689,0.9732,0.9763,0.9603],[0.9756,0.9752,0.9803,0.9779,0.9788,0.9755],[0.9831,0.9817,0.9814,0.9745,0.9741,0.9825]
         ,[0.9854,0.9822,0.9804,0.9723,0.9800,0.9852],[0.9858,0.9812,0.9799,0.9716,0.9576,0.9837]]

# Convert data to Pandas DataFrame
df = pd.DataFrame(data, columns=['Exit 1', 'Exit 2', 'Exit 3', 'Exit 4', 'Exit 5','Mixed Exit'],
                  index=['0 dB', '5 dB', '10 dB', '15 dB','20 dB'])
df2 = pd.DataFrame(data2, columns=['Exit 1', 'Exit 2', 'Exit 3', 'Exit 4', 'Exit 5','Mixed Exit'],
                  index=['0 dB', '5 dB', '10 dB', '15 dB','20 dB'])

# Define the color palette
custom_palette = {'Exit 1': [0.6, 0.8, 1.], 'Exit 2': [0.5, 0.7, 1.], 'Exit 3': [0.4, 0.6, 1.],
                  'Exit 4': [0.3, 0.5, 1.], 'Exit 5': [0.2, 0.4, 1.], 'Mixed Exit': [0.5, 0.5, 0.5]}

# Pivot the data to prepare for plotting
df_pivot = df.stack().reset_index()
df_pivot.columns = ['dataset', 'classifier', 'auroc']

df_pivot2 = df2.stack().reset_index()
df_pivot2.columns = ['dataset', 'classifier', 'auroc']
# Create a bar plot
fig, axs = plt.subplots(2, 1, figsize=(40, 20), sharey=True)
ax1 = sns.barplot(x=df_pivot['dataset'], y=df_pivot['auroc'], hue=df_pivot['classifier'], palette=custom_palette,ax=axs[0])
ax2 = sns.barplot(x=df_pivot2['dataset'], y=df_pivot2['auroc'], hue=df_pivot2['classifier'], palette=custom_palette,ax=axs[1])


# Set y-axis limits
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)

# Add legend
handles, labels = ax1.get_legend_handles_labels()
legend = ax1.legend(handles[:6], labels[:6], title='Exit Index', bbox_to_anchor=(1, 1), loc='upper left', )

ax2.get_legend().remove()
# legend2 = ax2.legend(handles[:6], labels[:6], title='Exit Index', bbox_to_anchor=(1, 1), loc='upper left', )
# Add axis labels and title

ax1.set_ylabel('AUROC')
ax1.set_ylim(0.9, 1.0)
ax1.set_xticklabels(df.index)
ax1.set_title('AUROC by SNR and Exit Index, Unknown Dataset: Imagenet Resize')

ax2.set_xlabel('SNR')
ax2.set_ylabel('AUROC')
ax2.set_ylim(0.9, 1.0)
ax2.set_xticklabels(df.index)
ax2.set_title('AUROC by SNR and Exit Index, Unknown Dataset: LSUN Resize')

# Show the plot
# plt.savefig("a.eps")
plt.show()
fig.savefig(f'dynmaic.eps',format='eps',dpi=300)