import numpy as np
import matplotlib.pyplot as plt



y_exct = np.load('y_true.npy')                                          

#print(y_exct.shape)
y = np.load('y_obs.npy')                                                

y_e = np.load('assimilated_ensemble.npy')
y_obs_alltime = np.load('simualated_all_time_obs.npy') 

print(y_obs_alltime.shape[0]) # total ensemble member 
y_avg_obs_alltime = (1/y_obs_alltime.shape[0])*(y_obs_alltime.sum(axis =0))
print(y_avg_obs_alltime.shape)



n_ensemble = np.shape(y_e)[0]  

y_e_tr_obs = np.transpose(y_e, (1,0,2))# use while plotiing against N_obs
                     
y_alltime = np.transpose(y_obs_alltime, (1,0,2))

y_e_trans_spatial = np.transpose(y_obs_alltime, (2,1,0)) # use while plotiing against X


y_exct_alltime = np.load('y_true_alltime.npy') 

y_obs_RB = np.zeros((y_exct_alltime.shape[0]))
y_obs_EME = np.zeros((y_exct_alltime.shape[0]))
print(y_exct_alltime.shape)
for i in range(y_exct_alltime.shape[0]):
    for j in range((y_obs_alltime.shape[0])):
        y_obs_EME[i] += (np.linalg.norm(y_exct_alltime[i,:]-y_obs_alltime[j,i,:])/ np.linalg.norm(y_exct_alltime[i,:]))
        y_obs_EME[i] /= y_obs_alltime.shape[0]
    y_obs_RB[i] = np.linalg.norm(y_exct_alltime[i,:]-y_avg_obs_alltime[i,:])/ np.linalg.norm(y_exct_alltime[i,:])


y_exct_alltime_trans_spatial = np.transpose(y_exct_alltime)

y_exct_noisy_alltime = np.load('y_obs_alltime.npy') 
y_exct_noisy_alltime_trans_spatial = np.transpose(y_exct_noisy_alltime)
#print(y_e_trans_spatial.shape)


xi =30
ensemble = 10
N = 1

y_e_mean_obs = np.mean(y_alltime[:,:,xi], axis=1)

# plt.plot(y_alltime[:,:,xi], 'y-')
# plt.plot(y_exct_alltime[:,xi], 'r-', label='true soln')
# plt.plot(y_exct_noisy_alltime[:,xi], '-o')
# plt.plot(y_e_mean_obs, 'b-', label = 'ensemble mean')
# plt.plot(y_e_trans_spatial[:,N, :], '-y')
# plt.plot(y_exct_alltime_trans_spatial[:,N], 'b-', label='true soln')
# plt.plot(y_exct_noisy_alltime_trans_spatial[:,N], '-o', label='noisy data')

#plt.plot(y_obs_RB, 'b-')
plt.plot(y_obs_EME, 'r-')
plt.title('EME for all weather stations')
plt.xlabel("time")
# plt.ylabel("velocity")
# plt.title('Ensemble trajectories')
# plt.legend()
plt.show()
        



