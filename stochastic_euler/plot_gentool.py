import numpy as np
import matplotlib.pyplot as plt


u_exact = np.load('u_true_data.npy')
u_vel = np.load('u_obs_data.npy') 
u_e = np.load('Velocity_ensemble.npy')
print(u_e.shape)

# print(u_exact)

#plt.plot(u_exact)

plt.plot(u_exact[:,19,0], 'r-',label='exact')
plt.plot(u_vel[:,19,0], '.', label='noisy data')
plt.legend()
plt.show()
