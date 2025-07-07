import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


RB_err_bs = np.load('2000_error_RB_bs.npy')
RB_err_temp = np.load('2000_error_RB_tempjitt.npy')
RB_err_nudge = np.load('1500_error_RB_nudge.npy')

N_t = 1500
N_df = 100

RB_err_bs_df = pd.DataFrame(RB_err_bs[:N_t])
RB_err_bs_roll =RB_err_bs_df.rolling(N_df).mean()

RB_err_temp_df = pd.DataFrame(RB_err_temp[:N_t])
RB_err_temp_roll =RB_err_temp_df.rolling(N_df).mean()

RB_err_nudge_df = pd.DataFrame(RB_err_nudge[:N_t])
RB_err_nudge_roll =RB_err_nudge_df.rolling(N_df).mean()


plt.plot(RB_err_bs_roll,  linestyle='-', color='r', markersize=6, linewidth=2,  label='Bootstrap')
plt.plot(RB_err_temp_roll,   linestyle='-', color='m', markersize=6, linewidth=2, label='Temp+jitt')
plt.plot(RB_err_nudge_roll,  linestyle='-', color='g', markersize=6, linewidth=2, label='Nudge only')
plt.title('Error Comparison RB')
plt.xlabel('DA steps')
plt.legend()
plt.show()


ERE_err_bs = np.load('2000_error_ERE_bs.npy')
ERE_err_temp = np.load('2000_error_ERE_tempjitt.npy')
ERE_err_nudge = np.load('1500_error_ERE_nudge.npy')


ERE_err_bs_df = pd.DataFrame(ERE_err_bs[:N_t])
ERE_err_bs_roll =ERE_err_bs_df.rolling(N_df).mean()

ERE_err_temp_df = pd.DataFrame(ERE_err_temp[:N_t])
ERE_err_temp_roll =ERE_err_temp_df.rolling(N_df).mean()

ERE_err_nudge_df = pd.DataFrame(ERE_err_nudge[:N_t])
ERE_err_nudge_roll =ERE_err_nudge_df.rolling(N_df).mean()

plt.plot(ERE_err_bs_roll,  linestyle='-', color='r', markersize=6, linewidth=2,  label='Bootstrap')
plt.plot(ERE_err_temp_roll,   linestyle='-', color='m', markersize=6, linewidth=2, label='Temp+jitt')
plt.plot(ERE_err_nudge_roll,  linestyle='-', color='g', markersize=6, linewidth=2, label='Nudge only')
plt.title('Error Comparison RMSE')
plt.xlabel('DA steps')
plt.legend()
plt.show()

