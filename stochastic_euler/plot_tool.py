import numpy as np

import matplotlib.pyplot as plt

u_e = np.load('Velocity_ensemble.npy')
print(u_e.shape)
u_exact = np.load('u_true_data.npy')
print(u_exact.shape)
u_obs_alltime = np.load('Velocity simualated_all_time.npy')

n_ensemble = np.shape(u_e)[0]

u_alltime = np.transpose(u_obs_alltime, (1,0,2,3))

#print(u_e.shape)

# Against time plot
u_e_time = np.transpose(u_e, (1,0,2,3))
u_e_x = np.transpose(u_e, (2,0,1,3))
u_exact_x = np.transpose(u_exact, (1,0,2))

print(u_e_x.shape)
print(u_exact_x.shape)
# Against DA steps
xi = 19
# plt.plot(u_e_time[:,:,xi,1])
# plt.plot(u_exact[:, xi,1], 'b-')

# Against weather station
N = 9
plt.plot(u_e_x[:,  :, N, 1], 'y-')
plt.plot(u_exact_x[:,N, 1], 'b-')

plt.show()