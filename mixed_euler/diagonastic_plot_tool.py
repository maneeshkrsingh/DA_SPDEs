import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

temp_step = np.load('new_tempering.npy')
print(temp_step[-1])

plt.hist(temp_step[-1], bins=10, edgecolor='black')
plt.show()