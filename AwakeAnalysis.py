'''
1. Compare power before, during and immediately after the light stimulus
2. Hilbert analysis at different layers to determine if the phase of gamma is different

'''

import csv
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from TimeFreq_plot import Run_spectrogram
from CSD import CSD_calc
import seaborn as sns
from scipy.signal import find_peaks
import scipy.stats as stats

'''
1. Extract LFP data and downsample to 200Hz (i.e. 5ms)
'''

# Import data
col_list = ["timestamps","timesamples","pri_0","pri_1","pri_2","pri_3","pri_4","pri_5","pri_6","pri_7","pri_8",
                "pri_9","pri_10","pri_11","pri_12","pri_13","pri_14","pri_15","pri_16","pri_17","pri_18","pri_19",
                    "pri_20","pri_21","pri_22","pri_23","pri_24","pri_25","pri_26","pri_27","pri_28","pri_29","pri_30",
                        "pri_31","din_2"]

df = pd.read_csv("D:\Tim's data\Data\ChR2DRD_awake_Mouse2_sbpro_5__uid1221-19-16-35.csv", usecols=col_list)

Time = df["timesamples"].to_numpy()
Time = Time[0:-1:50]
LightPulse = df["din_2"].to_numpy()
LightPulse = LightPulse[0:-1:50]
V1 = df["pri_7"].to_numpy()
V1 = V1[0:-1:50]

dt = Time[2]-Time[1]
print('Time step = ',dt)
print('Sample rate = ',1/dt)

StimStart_i = np.where(np.diff(LightPulse)==2)
print('No. of light pulses = ',len(StimStart_i[0]))

######################################################
# Power spectrum parameters
######################################################
srate = 1/dt
nyq = 0.5 * srate
freq = np.arange(0, (srate / 2) + (srate / (int(1/dt)-20)), srate / (int(1/dt)-20))
print('Freq length ',len(freq))
####################################################
def Spectrum(V_win):
	LFP_win = V_win - np.mean(V_win)
	# Windowing if you want
	w = np.hanning(len(V_win))
	LFP_win = w * LFP_win
	# Calculate power spectrum for window
	Fs = srate
	N = len(LFP_win)
	xdft = np.fft.fft(LFP_win)
	xdft = xdft[0:int((N / 2) + 1)]
	psdx = (1 / (Fs * N)) * np.abs(xdft) ** 2
	freq = np.arange(0, (Fs / 2) + Fs / N, Fs / N)
	Pow = psdx
	Pow = np.zeros((len(freq), 1))
	for j in range(0, 200):
		Pow[j] = psdx[j]
	return Pow, freq

######################################################
######################################################

Pulse_V = np.zeros((len(StimStart_i[0]),int(1/dt)-20))
PrePulse_V = np.zeros((len(StimStart_i[0]),int(1/dt)-20))
PostPulse_V = np.zeros((len(StimStart_i[0]),int(1/dt)-20))

for i in range(len(StimStart_i[0])):
    Pulse_V[i,:] = V1[StimStart_i[0][i]+20:StimStart_i[0][i]+int(1/dt)]
    PrePulse_V[i,:] = V1[StimStart_i[0][i]-int(1/dt):StimStart_i[0][i]-20]
    PostPulse_V[i,:] = V1[StimStart_i[0][i]+int(1/dt)+20:StimStart_i[0][i]+int(1/dt)+int(1/dt)]

T_Win = np.arange(0,(1-21*dt),dt)
Pulse_Pow = np.zeros((len(StimStart_i[0]),len(freq)))
PrePulse_Pow = np.zeros((len(StimStart_i[0]),len(freq)))
PostPulse_Pow = np.zeros((len(StimStart_i[0]),len(freq)))

for i in range(len(StimStart_i[0])):
    Pow, freq = Spectrum(Pulse_V[i,:])
    Pulse_Pow[i,:] = np.squeeze(Pow)
    Pow, freq = Spectrum(PrePulse_V[i, :])
    PrePulse_Pow[i, :] = np.squeeze(Pow)
    Pow, freq = Spectrum(PostPulse_V[i, :])
    PostPulse_Pow[i, :] = np.squeeze(Pow)
    '''
    Fig, axs = plt.subplots(3,1,sharex=True)
    axs[0].plot(T_Win,PrePulse_V[i,:])
    axs[1].plot(T_Win, Pulse_V[i, :])
    axs[2].plot(T_Win, PostPulse_V[i, :])
    '''

Pulse_Pow_SEM = stats.sem(Pulse_Pow,axis=0)
PrePulse_Pow_SEM = stats.sem(PrePulse_Pow,axis=0)
PostPulse_Pow_SEM = stats.sem(PostPulse_Pow,axis=0)


'''for i in range(len(StimStart_i[0])):
    plt.plot(freq,PrePulse_Pow[i,:],color = 'blue',linewidth = 0.2, alpha = 0.2)
    plt.plot(freq, Pulse_Pow[i,:], color='green', linewidth=0.2, alpha = 0.2)
    plt.plot(freq, PostPulse_Pow[i,:], color='red', linewidth=0.2, alpha = 0.2)
'''
print('Shape test ',np.shape(np.add(PrePulse_Pow,PrePulse_Pow_SEM)))
plt.fill_between(freq,np.add(np.mean(PrePulse_Pow,axis=0),PrePulse_Pow_SEM),np.subtract(np.mean(PrePulse_Pow,axis=0),PrePulse_Pow_SEM),color = 'blue',alpha = 0.2)
plt.plot(freq,np.mean(PrePulse_Pow,axis=0),color = 'blue',linewidth = 2, zorder = -1, label = 'Pre-pulse')
plt.fill_between(freq,np.add(np.mean(Pulse_Pow,axis=0),Pulse_Pow_SEM),np.subtract(np.mean(Pulse_Pow,axis=0),Pulse_Pow_SEM),color = 'green',alpha = 0.2)
plt.plot(freq,np.mean(Pulse_Pow,axis=0),color = 'green',linewidth = 2, zorder = -1, label = 'Pulse')
plt.fill_between(freq,np.add(np.mean(PostPulse_Pow,axis=0),PostPulse_Pow_SEM),np.subtract(np.mean(PostPulse_Pow,axis=0),PostPulse_Pow_SEM),color = 'red',alpha = 0.2)
plt.plot(freq,np.mean(PostPulse_Pow,axis=0),color = 'red',linewidth = 2, zorder = -1, label = 'Post-pulse')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0,150)
plt.legend()
plt.show()

# Find when light pulses occur
PulseTimes_i = np.where(LightPulse==2)

Fig, axs = plt.subplots(2,1,sharex = True)
axs[0].plot(Time[0:100000],V1[0:100000])
axs[0].scatter(Time[StimStart_i[0][0:10]+10],V1[StimStart_i[0][0:10]+10],marker = ',',color='red')
axs[1].plot(Time[0:100000],LightPulse[0:100000])
axs[1].scatter(Time[StimStart_i[0][0:10]],LightPulse[StimStart_i[0][0:10]],marker = ',',color='red')
plt.show()