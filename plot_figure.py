# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# x = ["Exit 1","Exit 2","Exit 3","Exit 4","Exit 5","Mixed Exit"]
# y = [0.9648,0.9582,0.9558,0.9482,0.9237,0.9714]
# fig = sns.barplot(x, y,color=)
# fig.set_title('PSNR = 20dB')
# fig.set_ylabel("Sequential")
# plt.show()
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = ["Exit 1", "Exit 2", "Exit 3", "Exit 4", "Exit 5", "Mixed Exit"]
y = [0.9215, 0.91504203, 0.9342796499999999, 0.94248573, 0.9478838300000002, 0.928395805]

data = {'Exit': x, 'Value': y}
df = pd.DataFrame(data)

# Calculate the number of columns
num_columns = len(df)

# Set the starting color for the blue shades
start_color = np.array([0.6, 0.8, 1])

# Set the ending color for the blue shades (blue)
end_color = np.array([0.2, 0.4, 1])

# Create a list to hold the colors
column_colors = []

# Generate the gradually darker blue colors
for i in range(num_columns - 1):
    color = start_color + (end_color - start_color) * i / (num_columns - 2)
    column_colors.append(color)

# Add the gray color for the final column
column_colors.append([0.5, 0.5, 0.5])

# Create the custom color map
custom_cmap = dict(zip(df['Exit'], column_colors))
print(custom_cmap)
# Plot the data
fig, ax = plt.subplots()
ax.set_title('PSNR = 0dB, Unknown Dataset = Imagenet resize')
ax.set_ylim(0.9, 1.0)
sns.barplot(x='Exit', y='Value', data=df, palette=custom_cmap, ax=ax)
plt.show()
