from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# particle_init =  np.load('particle_in.npy')



# # print(particle_init.shape)

# plt.plot(particle_init[:,:90], 'y-')
# plt.plot(particle_init[:,90:],  'g-', label='truth initilisation')
# plt.xlabel('mesh points')

# plt.legend()
# plt.show()

# quit()

y_exact = np.load('y_true.npy')
y_obs = np.load('y_obs.npy')
y_exact_time = np.transpose(y_exact, (1,0))
y_obs_time = np.transpose(y_obs, (1,0))
print('true',y_exact.shape)


y_e_bs = np.load('2000_bs_ensemble.npy') 
y_e_bs_time = np.transpose(y_e_bs, (1,0,2)) # (Nobs,  particles,  spatial) 
y_e_bs_x = np.transpose(y_e_bs, (2,0,1)) # (spatial, particles, Nobs)

print('y_e_bs_x', y_e_bs_x.shape) 


y_e_temp = np.load('2000_tempjitt_ensemble.npy')
print('ensemble', y_e_temp.shape) 
y_e_temp_time = np.transpose(y_e_temp, (1,0,2))
y_e_temp_x = np.transpose(y_e_temp, (2,0,1))
print(y_e_temp_time.shape)

# load all nudge
y_e_nudge_250 = np.load('250_nudge_only_ensemble.npy')
y_e_nudge_500 = np.load('500_nudge_only_ensemble.npy')
y_e_nudge_750 = np.load('750_nudge_only_ensemble.npy')
y_e_nudge_1000 = np.load('1000_nudge_only_ensemble.npy')
y_e_nudge_1250 = np.load('1250_nudge_only_ensemble.npy')
y_e_nudge_1500 = np.load('1500_nudge_only_ensemble.npy')

y_e_nudge = np.concatenate((y_e_nudge_250, y_e_nudge_500, y_e_nudge_750, y_e_nudge_1000, y_e_nudge_1250, y_e_nudge_1500), axis=1)
print('ensemble nudge', y_e_nudge.shape) 
y_e_nudge_time = np.transpose(y_e_nudge, (1,0,2))
y_e_nudge_x = np.transpose(y_e_nudge, (2,0,1))

# mean over ensemble
y_ensemble_temp_mean_time = np.mean(y_e_temp_time, axis=1)
y_ensemble_bs_mean_time = np.mean(y_e_bs_time, axis=1)
y_ensemble_nudge_mean_time = np.mean(y_e_nudge_time, axis=1)


# std over ensmeble
y_ensemble_bs_std_time = np.std(y_e_bs_time, axis=1)
y_ensemble_temp_std_time = np.std(y_e_temp_time, axis=1)
y_ensemble_nudge_std_time = np.std(y_e_nudge_time, axis=1)

# # # # # #Against DA steps
# N = 249
# plt.plot(y_e_bs_x[:, :, N], 'r-')
# plt.plot(y_e_temp_x[:, :, N], 'y-')
# plt.plot(y_e_nudge_x[:, :, N], 'g-')
# plt.plot(y_exact[N, :], 'b-', label='exact')
# #plt.plot(y_obs[N, :], 'o', label='obs')
# # plt.xlabel("observation points")
# # plt.legend()
# # plt.title('DA Step '+str(N))
# plt.show()
################################################  bs #################################################

# # # # # # observation step
xi = 3
N_start = 975
N_t = 1000
# Plot individual ensemble members in gray
plt.plot(y_e_bs_time[N_start:N_t, :, xi], color="gray", alpha=0.3)
# Plot ensemble mean in black
plt.plot(y_ensemble_bs_mean_time[N_start:N_t, xi], color="blue", linestyle="-")
# Plot standard deviation bounds
plt.plot(y_ensemble_bs_mean_time[N_start:N_t, xi] + y_ensemble_bs_std_time[N_start:N_t, xi], color="black", linestyle="--")
plt.plot(y_ensemble_bs_mean_time[N_start:N_t, xi] - y_ensemble_bs_std_time[N_start:N_t, xi], color="black", linestyle="--")
plt.plot(y_ensemble_bs_mean_time[N_start:N_t, xi] + 2*y_ensemble_bs_std_time[N_start:N_t, xi], color="black", linestyle=":")
plt.plot(y_ensemble_bs_mean_time[N_start:N_t, xi] - 2*y_ensemble_bs_std_time[N_start:N_t, xi], color="black", linestyle=":")
#plot truth
plt.plot(y_exact[N_start:N_t, xi], color="red", linestyle="-", linewidth=2)

# Labels and legend
plt.xlabel("DA Steps (975-1000)")
#plt.ylabel("Value")
plt.title(" particles  at index 3")
plt.legend()
plt.show()


##############################################  tempering ###############################################
# # # # # # observation step
xi = 3
N_start = 975
N_t = 1000
# Plot individual ensemble members in gray
plt.plot(y_e_temp_time[N_start:N_t, :, xi], color="gray", alpha=0.3)
# Plot ensemble mean in black
plt.plot(y_ensemble_temp_mean_time[N_start:N_t, xi], color="blue", linestyle="-")
# Plot standard deviation bounds
plt.plot(y_ensemble_temp_mean_time[N_start:N_t, xi] + y_ensemble_temp_std_time[N_start:N_t, xi], color="black", linestyle="--")
plt.plot(y_ensemble_temp_mean_time[N_start:N_t, xi] - y_ensemble_temp_std_time[N_start:N_t, xi], color="black", linestyle="--")
plt.plot(y_ensemble_temp_mean_time[N_start:N_t, xi] + 2*y_ensemble_temp_std_time[N_start:N_t, xi], color="black", linestyle=":")
plt.plot(y_ensemble_temp_mean_time[N_start:N_t, xi] - 2*y_ensemble_temp_std_time[N_start:N_t, xi], color="black", linestyle=":")
#plot truth
plt.plot(y_exact[N_start:N_t, xi], color="red", linestyle="-", linewidth=2)

# Labels and legend
plt.xlabel("DA Steps (975-1000)")
#plt.ylabel("Value")
plt.title(" particles  at index 3")
plt.legend()
plt.show()


#######################################  nudging ################################

# # # # # # observation step
xi = 3
N_start = 975
N_t = 1000
# Plot individual ensemble members in gray
plt.plot(y_e_nudge_time[N_start:N_t, :, xi], color="gray", alpha=0.3)
# Plot ensemble mean in black
plt.plot(y_ensemble_nudge_mean_time[N_start:N_t, xi], color="blue", linestyle="-")
# Plot standard deviation bounds
plt.plot(y_ensemble_nudge_mean_time[N_start:N_t, xi] + y_ensemble_nudge_std_time[N_start:N_t, xi], color="black", linestyle="--")
plt.plot(y_ensemble_nudge_mean_time[N_start:N_t, xi] - y_ensemble_nudge_std_time[N_start:N_t, xi], color="black", linestyle="--")
plt.plot(y_ensemble_nudge_mean_time[N_start:N_t, xi] + 2*y_ensemble_nudge_std_time[N_start:N_t, xi], color="black", linestyle=":")
plt.plot(y_ensemble_nudge_mean_time[N_start:N_t, xi] - 2*y_ensemble_nudge_std_time[N_start:N_t, xi], color="black", linestyle=":")
#plot truth
plt.plot(y_exact[N_start:N_t, xi], color="red", linestyle="-", linewidth=2)

# Labels and legend
plt.xlabel("DA Steps (975-1000)")
#plt.ylabel("Value")
plt.title(" particles  at index 3")
plt.legend()
plt.show()



