import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load energy data
energy_one = np.load("SFLT_peakanti_energy_one.npy")
energy_two = np.load("SFLT_peakanti_energy_two.npy")
total_energy = np.load("SFLT_peakanti_TotalEnergy.npy")

# Build DataFrame including total_energy
df = pd.DataFrame({
    'energy_one': energy_one,
    'energy_two': energy_two,
    'total_energy': total_energy
})

# Apply smoothing
window = 2
df['energy_one_smooth'] = df['energy_one'].rolling(window=window, center=True, min_periods=1).mean()
df['energy_two_smooth'] = df['energy_two'].rolling(window=window, center=True, min_periods=1).mean()
df['total_energy_smooth'] = df['total_energy'].rolling(window=window, center=True, min_periods=1).mean()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(df['energy_one_smooth'], color='b', linewidth=2, label=r'$E_1$')
plt.plot(df['energy_two_smooth'], color='g', linewidth=2, label=r'$E_2$')
plt.plot(df['total_energy_smooth'], color='orange', linewidth=2, label=r'$E_1 + E_2$')
plt.title("Energy Terms vs Time", fontsize=16, fontweight='bold')
plt.xlabel("Time Step", fontsize=13)
plt.ylabel("Energy", fontsize=13)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

