import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



y_exct = np.load('y_true.npy')                                          

#print(y_exct.shape)
y_exct_obs = np.load('y_obs.npy')                                                
print(y_exct_obs.shape)
y_e = np.load('Simplifiednudge_assimilated_ensemble.npy')
print(y_e.shape)
y_avg_e = (1/y_e.shape[0])*(y_e.sum(axis =0))


y_RB = np.zeros((y_exct.shape[0]))
y_EME = np.zeros((y_exct.shape[0]))
for i in range(y_exct.shape[0]):
    for j in range((y_e.shape[0])):
        y_EME[i] += (np.linalg.norm(y_exct[i,:]-y_e[j,i,:])/ np.linalg.norm(y_exct[i,:]))
    y_EME[i] /= y_e.shape[0]
    y_RB[i] = np.linalg.norm(y_exct[i,:]-y_avg_e[i,:])/ np.linalg.norm(y_exct[i,:])

y_RB_df = pd.DataFrame(y_RB)
y_RB_roll = y_RB_df.rolling(10).mean()
y_EME_df = pd.DataFrame(y_EME)
y_EME_roll = y_EME_df.rolling(10).mean()

# plt.plot(y_RB_roll, 'b-', label='EME')
# plt.plot(y_EME_roll, 'r-', label='RB')
# plt.title('EME/RB for all weather stations')
# plt.xlabel("DA steps")
# plt.ylabel("EME/RB")
# plt.legend()


y_obs_alltime = np.load('Simplifiednudge_simualated_all_time_obs.npy') 
y_obs_alltime_new = np.load('Simplifiednudge_new_simualated_all_time_obs.npy') 

#print(y_obs_alltime.shape[0]) # total ensemble member 
y_avg_obs_alltime = (1/y_obs_alltime.shape[0])*(y_obs_alltime.sum(axis =0))
y_avg_obs_alltime_new = (1/y_obs_alltime.shape[0])*(y_obs_alltime_new.sum(axis =0))
#print(y_avg_obs_alltime.shape)



n_ensemble = np.shape(y_e)[0]  

y_ensemble_N = np.transpose(y_e, (1,0,2))# use while plotiing against N_obs
print(y_ensemble_N.shape)     
xi = 20
# plt.plot(y_ensemble_N[:,:,xi], 'y-')
# plt.plot(y_exct[:,xi], 'r-', label='true soln')
# plt.plot(y_exct_obs[:,xi], 'b-', label='noisy data')

y_ensemble_X = np.transpose(y_e, (2,1,0))# use while plotiing against X
N = 10
print(y_ensemble_X.shape)    
# plt.plot(y_ensemble_X[:,N, :])
# plt.plot(y_exct[N,:], 'b-', label='true soln')
# plt.plot(y_exct_obs[N,:], '-o', label='noisy data')



y_alltime = np.transpose(y_obs_alltime, (1,0,2))
y_alltime_new = np.transpose(y_obs_alltime_new, (1,0,2))

y_e_trans_spatial = np.transpose(y_obs_alltime, (2,1,0)) # use while plotiing against X

y_e_trans_spatial_new = np.transpose(y_obs_alltime_new, (2,1,0)) # use while plotiing against X


y_exct_alltime = np.load('y_true_alltime.npy') 

y_obs_RB = np.zeros((y_exct_alltime.shape[0]))
y_obs_EME = np.zeros((y_exct_alltime.shape[0]))
#print(y_exct_alltime.shape)
for i in range(y_exct_alltime.shape[0]):
    for j in range((y_obs_alltime.shape[0])):
        y_obs_EME[i] += (np.linalg.norm(y_exct_alltime[i,:]-y_obs_alltime_new[j,i,:])/ np.linalg.norm(y_exct_alltime[i,:]))
    y_obs_EME[i] /= y_obs_alltime.shape[0]
    y_obs_RB[i] = np.linalg.norm(y_exct_alltime[i,:]-y_avg_obs_alltime_new[i,:])/ np.linalg.norm(y_exct_alltime[i,:])
y_obs_RB_df = pd.DataFrame(y_obs_RB)
y_obs_RB_roll = y_obs_RB_df.rolling(10).mean()
y_obs_EME_df = pd.DataFrame(y_obs_EME)
y_obs_EME_roll = y_obs_EME_df.rolling(10).mean()

y_exct_alltime_trans_spatial = np.transpose(y_exct_alltime)

y_exct_noisy_alltime = np.load('y_obs_alltime.npy') 
y_exct_noisy_alltime_trans_spatial = np.transpose(y_exct_noisy_alltime)
#print(y_e_trans_spatial.shape)


xi =7
ensemble = 10
N =0

y_e_mean_obs = np.mean(y_alltime[:,:,xi], axis=1)
y_e_mean_obs_df = pd.DataFrame(y_e_mean_obs)
y_e_mean_obs_roll = y_e_mean_obs_df.rolling(5).mean()

plt.plot(y_exct_alltime[:,xi], 'b-', label='Truth')
#plt.plot(y_alltime[:,:,xi], 'y-')
plt.plot(y_e_mean_obs_roll, 'r-', label = 'Ensemble mean')

#plt.plot(y_alltime_new[:,:,xi], 'r-')
#plt.plot(y_exct_noisy_alltime[:,xi], 'r-', label='noisy data')

# plt.plot(y_exct_alltime_trans_spatial[:,N], '-*', label='true soln')
# plt.plot(y_e_trans_spatial[:,N, :])
# plt.plot(y_exct_noisy_alltime_trans_spatial[:,N], '-o', label='noisy data')

# #plt.plot(y_e_trans_spatial_new[:,N, :], '-y')



# plt.plot(y_obs_RB_roll, 'b-')
# plt.plot(y_obs_EME_roll, 'r-')
#plt.plot(y_obs_EME, 'r-')
#plt.title('EME for all weather stations')
plt.xlabel("x")
plt.ylabel("velocity")
plt.title('Ensemble trajectories')
plt.legend()
plt.show()
        



