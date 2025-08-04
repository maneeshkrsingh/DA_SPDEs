from ctypes import sizeof
from fileinput import filename
from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.output import VTKFile
from nudging.models.stochastic_Camassa_Holm import Camsholm


import os

# Output directory
# output_dir = "../../DA_Results/SCH/SFLT/NewGaussianwide/SCH_particles/"
output_dir = "../../DA_Results/SCH/SALT/Zerovelocity/SCH_particles/"
os.makedirs(output_dir, exist_ok=True)



"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
# Simulation parameters
N_obs = 4*4000
Nensemble = 5
nsteps = 1
xpoints = 5000
Ld = 40.0
n = 5000
alpha = 1.0
alphasq = Constant(alpha ** 2)
epsilon = Constant(0.01)
peak_width = 1 / 6
deltax = Ld / n

# Model setup
model = Camsholm(n, Ld, nsteps, xpoints, seed=12345, lambdas=False, salt=True)
model.setup()
x, = SpatialCoordinate(model.mesh)
# ic dictionary
ic_dict = {'two_peaks': (0.2*2/(exp(x-403./15.*40./Ld) + exp(-x+403./15.*40./Ld))
                             + 0.5*2/(exp(x-203./15.*40./Ld)+exp(-x+203./15.*40./Ld))),
               'gaussian': 0.5*exp(-((x-10.)/2.)**2),
               'gaussian_narrow': 0.5*exp(-((x-10.)/1.)**2),
               'gaussian_wide': 0.5*exp(-((x-10.)/3.)**2),
               'peakon': conditional(x < Ld/2., exp((x-Ld/2)/sqrt(alphasq)), exp(-(x-Ld/2)/sqrt(alphasq))),
               'one_peak': 0.5*2/(exp(x-203./15.*40./Ld)+exp(-x+203./15.*40./Ld)),
               'proper_peak': 0.5*2/(exp(x-Ld/4)+exp(-x+Ld/4)),
               'new_peak': 0.5*2/(exp((x-Ld/4)/peak_width)+exp((-x+Ld/4)/peak_width)),
               'peakon_antipeakon':exp(-(1/alpha) * abs(x - Ld/4.)) -  exp(-(1/alpha) * abs(x - 3*Ld/4.)),
               'flat': Constant(2*pi**2/(9*40**2)),
               'fast_flat': Constant(0.1),
               'coshes': Constant(2000)*cosh((2000**0.5/2)*(x-0.75))**(-2)+Constant(1000)*cosh(1000**0.5/2*(x-0.25))**(-2),
               'd_peakon':exp(-sqrt((x-Ld/2)**2 + epsilon * deltax ** 2) / sqrt(alphasq)),
               'zero': Constant(0.0),
               'two_peakons': conditional(x < Ld/4, exp((x-Ld/4)/sqrt(alphasq)) - exp(-(x+Ld/4)/sqrt(alphasq)),
                                          conditional(x < 3*Ld/4, exp(-(x-Ld/4)/sqrt(alphasq)) - exp((x-3*Ld/4)/sqrt(alphasq)),
                                                      exp((x-5*Ld/4)/sqrt(alphasq)) - exp(-(x-3*Ld/4)/sqrt(alphasq)))),
                'twin_wide_gaussian': exp(-((x-10.)/3.)**2) + 0.5*exp(-((x-30.)/3.)**2),
               'twin_peakons': conditional(x < Ld/4, exp((x-Ld/4)/sqrt(alphasq)) + 0.5* exp((x-Ld/2)/sqrt(alphasq)),
                                           conditional(x < Ld/2, exp(-(x-Ld/4)/sqrt(alphasq)) + 0.5* exp((x-Ld/2)/sqrt(alphasq)),
                                                       conditional(x < 3*Ld/4, exp(-(x-Ld/4)/sqrt(alphasq)) + 0.5 * exp(-(x-Ld/2)/sqrt(alphasq)),
                                                                   exp((x-5*Ld/4)/sqrt(alphasq)) + 0.5 * exp(-(x-Ld/2)/sqrt(alphasq))))),
               'periodic_peakon': (conditional(x < Ld/2, 0.5 / (1 - exp(-Ld/sqrt(alphasq))) * (exp((x-Ld/2)/sqrt(alphasq))
                                                                                                + exp(-Ld/sqrt(alphasq))*exp(-(x-Ld/2)/sqrt(alphasq))),
                                                         0.5 / (1 - exp(-Ld/sqrt(alphasq))) * (exp(-(x-Ld/2)/sqrt(alphasq))
                                                                                               + exp(-Ld/sqrt(alphasq))*exp((x-Ld/2)/sqrt(alphasq))))),
               'cos_bell':conditional(x < Ld/4, (cos(pi*(x-Ld/8)/(2*Ld/8)))**2, 0.0),
               'antisymmetric': 1/(exp((x-Ld/4)/Ld)+exp((-x+Ld/4)/Ld)) - 1/(exp((Ld-x-Ld/4)/Ld)+exp((Ld+x+Ld/4)/Ld))}


u_ic = ic_dict['peakon_antipeakon']  # initial condition
X_truth = model.allocate()
_, u0 = X_truth[0].split()

# Use a linear function space on mesh vertices (assuming CG1)
V = FunctionSpace(model.mesh, "CG", 1)  
L = 40.0
N = V.dim()  # number of DOFs in V, should match len(Zall)

# Create x_pos assuming uniform periodic mesh
x_pos = np.linspace(0, L, N, endpoint=False)  # match length of solution vectors

peak_positions = []

print("Domain min x:", x_pos.min(), "max x:", x_pos.max())
print("Number of DOFs in function space:", N)

ndump = 100

for p in ProgressBar("Ensemble members").iter(range(Nensemble)):
    u0.interpolate(0.0)  # Set initial condition for each particle
    dumpn = 0  
    time_index = 0  
    particle_file = VTKFile(f"{output_dir}/particle_{p}.pvd")
    f = Function(V, name=f"particle_{p}")
	# --- Add this for energy storage ---
    energy_one = []
    energy_two = []
    total_energy = []  # List to store total energy values at each timestep
    for i in ProgressBar(f"Timestep {p}").iter(range(N_obs)):
        model.randomize(X_truth)
        model.run(X_truth, X_truth)

        _, z = X_truth[0].subfunctions
        if p ==0:
            E_1 = assemble((z*z)*dx)
            E_2 = assemble((z.dx(0)*z.dx(0))*dx)
		    # --- Save energy and time ---
            energy_one.append(E_1)
            energy_two.append(E_2)
            total_energy.append(E_1 + E_2)
        Zall = z.dat.data_ro[:]

        #print(f"Timestep {i}: Length of Zall = {len(Zall)}, Length of x_pos = {len(x_pos)}")

        # Find index of max (peak)
        idx_max = np.argmax(Zall)
        x_star = x_pos[idx_max]

        peak_positions.append((i, x_star))

        # Save data for visualization (optional)
        f.dat.data[:] = Zall

        # # Uncomment to save VTK every ndump steps
        dumpn += 1
        if dumpn == ndump:
            particle_file.write(f, time=time_index)
            time_index += 1
            dumpn = 0
# # After time loop, convert to numpy arrays
# energy_one = np.array(energy_one)
# energy_two = np.array(energy_two)
# # # Save energy arrays to disk
np.save("zeroSFLT_peakanti_energy_one.npy", np.array(energy_one))
np.save("zeroSFLT_peakanti_energy_two.npy", np.array(energy_two))
np.save("zeroSFLT_peakanti_TotalEnergy.npy",   np.array(total_energy))
# Convert peak positions list to array
peak_positions = np.array(peak_positions)
timesteps = peak_positions[:, 0]
x_star_vals = peak_positions[:, 1]

# Plot soliton peak trajectory (x vs t)
# plt.figure(figsize=(6, 4))
# plt.plot(x_star_vals, timesteps, marker='o', linestyle='-', color='black')
# plt.xlabel(r'Peak position $x^*(t)$')
# plt.ylabel(r'Time $t$')
# plt.title('Trajectory of soliton peak in space-time')
# plt.grid(True)
# plt.tight_layout()
# plt.show()



import matplotlib.cm as cm

plt.figure(figsize=(8, 5))

# Normalize timesteps for colormap
norm = plt.Normalize(timesteps.min(), timesteps.max())
cmap = cm.viridis

# Scatter plot with color gradient by time
sc = plt.scatter(x_star_vals, timesteps, c=timesteps, cmap=cmap, norm=norm, s=40, edgecolor='k', alpha=0.85, label='Peak position')

# Connect points with a smooth line
plt.plot(x_star_vals, timesteps, color='gray', linestyle='--', alpha=0.6)

plt.xlabel(r'Peak position $x^*(t)$', fontsize=14)
plt.ylabel(r'Time $t$', fontsize=14)
plt.title('Soliton Peak Trajectory in Space-Time', fontsize=16)
plt.colorbar(sc, label='Time step')
plt.grid(visible=True, linestyle=':', alpha=0.7)

# Add annotation for peak position range
xmin, xmax = np.min(x_star_vals), np.max(x_star_vals)
plt.text(xmax*0.5, timesteps.max()*0.9, f'Peak position range:\n[{xmin:.2f}, {xmax:.2f}]', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.7))

plt.tight_layout()
plt.show()

