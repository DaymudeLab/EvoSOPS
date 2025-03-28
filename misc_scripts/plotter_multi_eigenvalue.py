from cmcrameri.cm import batlow
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

# List of CSV file paths
file_paths = [
    'output/eigen_val/Agg_CMA_1_sizes_1_trials_gran_10_3744288131_EigenVal.csv',
    'output/eigen_val/Agg_CMA_1_sizes_1_trials_gran_10_4177721761_EigenVal.csv',
    'output/eigen_val/Agg_CMA_1_sizes_2_trials_gran_10_1791688383_EigenVal.csv',
    'output/eigen_val/Agg_CMA_1_sizes_2_trials_gran_10_3617115867_EigenVal.csv',
    'output/eigen_val/Agg_CMA_1_sizes_2_trials_gran_10_3731390385_EigenVal.csv',
    'output/eigen_val/Sep_CMA_1_sizes_2_trials_gran_10_1585878937_EigenVal.csv',
    'output/eigen_val/Sep_CMA_1_sizes_2_trials_gran_10_1769147479_EigenVal.csv'
]

plt.figure(figsize=(10, 6))  # Set figure size
colors = batlow(np.linspace(0, 1, len(file_paths)))  # Generate unique colors

# Process each file and plot
for i, file_path in enumerate(file_paths):
    if osp.exists(file_path):
        sums = []  # Store sum of each line
        with open(file_path, "r") as file:
            for line in file:
                numbers = list(map(float, line.strip().split(",")))  # Convert to floats
                sums.append(sum(numbers))  # Compute sum of line values
        
        # Plot each file's data
        plt.plot(sums, marker="o", linestyle="-", color=colors[i], label=f"File {i+1}")
    else:
        print(f"File {file_path} does not exist.")

# Customize plot
plt.xlabel("Line Index")
plt.ylabel("Sum of Eigenvalues")
plt.title("Multi-Line Plot of Sum of Eigenvalues per Line")
plt.xlim(0, 1000)  # Set x-axis limit
plt.ylim(0, 700)   # Set y-axis limit
plt.legend(title="Files", loc="best")  # Place legend automatically
plt.grid(True)

plt.show()



