# Script path: results/runtime/runtime_plot.py


## load dependencies - third party
import matplotlib.pyplot as plt
import numpy as np


## PLOT #1 - Runtime per Data Augmentation Strategy ##

# Data extracted from LaTeX table
strategies = [
    "RU", "RO", "WERCS", "GN", "SMOTER", "SMOGN",
    "WSMOTER", "DAVID", "KNNOR-REG", "CARTGen-IR"
]
runtimes = [
    0.027, 0.058, 0.002, 1.518, 5.147, 5.350,
    0.314, 24.077, 0.034, 0.183
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(strategies, runtimes, color='darkseagreen', edgecolor='black')
plt.yscale('log')  # Logarithmic scale for runtime
plt.ylabel(' Mean Runtime (seconds, log scale)')
plt.xlabel('Strategy')
#plt.title('Runtime per Data Augmentation Strategy')
plt.xticks(rotation=45)
plt.grid(axis='y', which='major', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Show the plot
plt.show()



## PLOT #2 - Runtime per Data Augmentation Strategy with Standard Deviation Bars ##

# Data from the LaTeX table
strategies = [
    "RU", "RO", "WERCS", "GN", "SMOTER", "SMOGN",
    "WSMOTER", "DAVID", "KNNOR-REG", "CARTGen-IR"
]
runtimes = np.array([
    0.027, 0.058, 0.002, 1.518, 5.147, 5.350,
    0.314, 24.077, 0.034, 0.183
])
std_devs = np.array([
    0.000, 0.000, 0.000, 0.008, 0.016, 0.019,
    0.003, 0.083, 0.005, 0.001
])

# Plot with error bars
plt.figure(figsize=(10, 6))
plt.bar(strategies, runtimes, yerr=std_devs, capsize=5, color='darkseagreen', edgecolor='black')
plt.yscale('log')
plt.ylabel(' Mean Runtime (seconds, log scale)')
plt.xlabel('Strategy')
#plt.title('Runtime per Data Augmentation Strategy')
plt.xticks(rotation=45)
plt.grid(axis='y', which='major', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Show the plot
plt.show()


## PLOT #3 - Runtime per Data Augmentation Strategy with Standard Deviation Bars (Dot Plot) ##

# Data
strategies = [
    "RU", "RO", "WERCS", "GN", "SMOTER", "SMOGN",
    "WSMOTER", "DAVID", "KNNOR-REG", "CARTGen-IR"
]
runtimes = np.array([
    0.027, 0.058, 0.002, 1.518, 5.147, 5.350,
    0.314, 24.077, 0.034, 0.183
])
std_devs = np.array([
    0.000, 0.000, 0.000, 0.008, 0.016, 0.019,
    0.003, 0.083, 0.005, 0.001
])

# Create figure
plt.figure(figsize=(10, 6))
x = np.arange(len(strategies))

# Dot plot with error bars
plt.errorbar(x, runtimes, yerr=std_devs, fmt='o', capsize=5, color='darkseagreen', ecolor='gray', elinewidth=1.2, markersize=8)

# Formatting
plt.yscale('log')
plt.xticks(x, strategies, rotation=45)
plt.ylabel('Runtime (seconds, log scale)')
plt.xlabel('Strategy')
#plt.title('Mean Runtime per Strategy (Dot Plot with Std. Dev)')
plt.grid(axis='y', which='major', linestyle='--', linewidth=0.7)
plt.tight_layout()

# Show plot
plt.show()