from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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

# Avg over ensemble
y_ensemble_temp_avg_time = np.mean(y_e_temp_time, axis=1)
y_ensemble_bs_avg_time = np.mean(y_e_bs_time, axis=1)
y_ensemble_nudge_avg_time = np.mean(y_e_nudge_time, axis=1)

# # # # RMSE ########################
N_rmse = 1500
y_rmse_bs = np.zeros(N_rmse)
y_rmse_temp = np.zeros(N_rmse)
y_rmse_nudge = np.zeros(N_rmse)

# all weather station
for i in range(N_rmse):
    for p in range(y_e_temp_time.shape[1]):
        y_rmse_bs[i] += sqrt(1/y_e_temp_time.shape[1])*np.linalg.norm(y_exact[i,:]-y_e_bs_time[i,p, :])/np.linalg.norm(y_exact[i,:])
        y_rmse_temp[i] += sqrt(1/y_e_temp_time.shape[1])*np.linalg.norm(y_exact[i, :]-y_e_temp_time[i,p, :])/np.linalg.norm(y_exact[i,:])
        y_rmse_nudge[i] += sqrt(1/y_e_temp_time.shape[1])*np.linalg.norm(y_exact[i, :]-y_e_nudge_time[i,p, :])/np.linalg.norm(y_exact[i,:])

N_df = 100

y_rmse_bs_df = pd.DataFrame(y_rmse_bs)
y_rmse_bs_roll =y_rmse_bs_df.rolling(N_df).mean()

y_rmse_temp_df = pd.DataFrame(y_rmse_temp)
y_rmse_temp_roll =y_rmse_temp_df.rolling(N_df).mean()

y_rmse_nudge_df = pd.DataFrame(y_rmse_nudge)
y_rmse_nudge_roll =y_rmse_nudge_df.rolling(N_df).mean()

plt.plot(y_rmse_bs_roll, 'r-', label='Bootstrap')
plt.plot(y_rmse_temp_roll, 'y-', label='Temp+jittering')
plt.plot(y_rmse_nudge_roll, 'g-', label='Nudge only')
plt.xlabel("DA steps ")
plt.title('RMSE ')
plt.legend()
plt.show()

# # # # Bias ########################
y_rb_bs = np.zeros(N_rmse)
y_rb_temp = np.zeros(N_rmse)
y_rb_nudge = np.zeros(N_rmse)

for i in range(N_rmse):
    for m in range(y_exact.shape[1]):
        y_rb_bs[i] += abs(y_exact[i, m]-y_ensemble_bs_avg_time[i, m])/(np.sum(abs(y_exact[i, :])))
        y_rb_temp[i] += abs(y_exact[i, m]-y_ensemble_temp_avg_time[i, m])/(np.sum(abs(y_exact[i, :])))
        y_rb_nudge[i] += abs(y_exact[i, m]-y_ensemble_nudge_avg_time[i, m])/(np.sum(abs(y_exact[i, :])))

y_rb_temp_df = pd.DataFrame(y_rb_temp)
y_rb_bs_df = pd.DataFrame(y_rb_bs)
y_rb_temp_roll =y_rb_temp_df.rolling(N_df).mean()
y_rb_bs_roll =y_rb_bs_df.rolling(N_df).mean()

y_rb_nudge_df = pd.DataFrame(y_rb_nudge)
y_rb_nudge_roll =y_rb_nudge_df.rolling(N_df).mean()

plt.plot(y_rb_bs_roll, 'r-', label='Bootstrap')
plt.plot(y_rb_temp_roll, 'y-', label='Temp+jittering')
plt.plot(y_rb_nudge_roll, 'g-', label='Nudge only')
plt.title('RB ')
plt.xlabel("DA steps ")
plt.legend()
plt.show()


#################### Relative l2 error #############################
y_Rl2_bs = np.zeros(N_rmse)
y_Rl2_temp = np.zeros(N_rmse)
y_Rl2_nudge = np.zeros(N_rmse)

for i in range(N_rmse):
    for p in range(y_e_temp_time.shape[1]):
        y_Rl2_bs[i] += (1/y_e_temp_time.shape[1])*np.linalg.norm(y_exact[i, :]-y_e_bs_time[i,p, :])/np.linalg.norm(y_exact[i, :])
        y_Rl2_temp[i] += (1/y_e_temp_time.shape[1])*np.linalg.norm(y_exact[i, :]-y_e_temp_time[i,p, :])/np.linalg.norm(y_exact[i, :])
        y_Rl2_nudge[i] += (1/y_e_temp_time.shape[1])*np.linalg.norm(y_exact[i, :]-y_e_nudge_time[i,p, :])/np.linalg.norm(y_exact[i, :])

y_Rl2_bs_df = pd.DataFrame(y_Rl2_bs)
y_Rl2_bs_roll =y_Rl2_bs_df.rolling(N_df).mean()

y_Rl2_temp_df = pd.DataFrame(y_Rl2_temp)
y_Rl2_temp_roll =y_Rl2_temp_df.rolling(N_df).mean()

y_Rl2_nudge_df = pd.DataFrame(y_Rl2_nudge)
y_Rl2_nudge_roll =y_Rl2_nudge_df.rolling(N_df).mean()



plt.plot(y_Rl2_bs_roll, 'r-', label='Bootstrap')
plt.plot(y_Rl2_temp_roll, 'y-', label='Temp+jittering')
plt.plot(y_Rl2_nudge_roll, 'g-', label='Nudge only')

plt.xlabel("DA steps ")
plt.title('Relative L2 error ')
plt.legend()
plt.show()
