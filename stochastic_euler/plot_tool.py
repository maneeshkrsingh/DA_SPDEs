import numpy as np

import matplotlib.pyplot as plt

u_e = np.load('Velocity_ensemble.npy')
u_exact = np.load('u_true_data.npy')
u_obs_alltime = np.load('Velocity simualated_all_time.npy')

n_ensemble = np.shape(u_e)[0]

u_alltime = np.transpose(u_obs_alltime, (1,0,2,3))

print(u_alltime.shape)

xi = 21

plt.plot(u_alltime[:,:,xi,1], 'g-')
plt.show()