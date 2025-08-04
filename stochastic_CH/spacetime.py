import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSVs
df_sflt = pd.read_csv("../../DA_Results/SCH/SFLT/particle1.csv")
df_salt = pd.read_csv("../../DA_Results/SCH/SFLT/particle_SALT_1.csv")

# Columns
time_col = 'TimeStep'
space_col = 'arc_length'
field_col = 'particle_0'

# Pivot to (time × space) matrices
U_sflt = df_sflt.pivot(index=time_col, columns=space_col, values=field_col)
U_salt = df_salt.pivot(index=time_col, columns=space_col, values=field_col)

# Shared color scale limits
vmin = min(U_sflt.min().min(), U_salt.min().min())
vmax = max(U_sflt.max().max(), U_salt.max().max())

# Create figure with GridSpec to control colorbar layout
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, gridspec_kw={'width_ratios': [1, 1]})

# Plot SFLT
im0 = axes[0].pcolormesh(U_sflt.columns, U_sflt.index, U_sflt.values,
                         shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[0].set_title("SFLT")
axes[0].set_xlabel(r'$x$')
axes[0].set_ylabel(r'$t$')

# Plot SALT
im1 = axes[1].pcolormesh(U_salt.columns, U_salt.index, U_salt.values,
                         shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[1].set_title("SALT")
axes[1].set_xlabel(r'$x$')

# Create single colorbar to the right
cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.046, pad=0.04)
cbar.set_label(field_col)

# Final layout
plt.suptitle(r'Space-time comparison of $u(x, t)$: SFLT vs SALT', fontsize=12)
#plt.tight_layout(rect=[0, 0, 1, 0.9995])
plt.show()
