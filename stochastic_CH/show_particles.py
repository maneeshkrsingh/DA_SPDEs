import numpy as np
import matplotlib.pyplot as plt
import os

particles = np.load("../../DA_Results/SCH/particles_all_x.npy")
plt.plot(particles[0, 3900, :], label='ensemble 1')
plt.plot(particles[1, 3900, :], label='ensemble 2')
plt.plot(particles[2, 3900, :], label='ensemble 3')
plt.legend()
plt.show()