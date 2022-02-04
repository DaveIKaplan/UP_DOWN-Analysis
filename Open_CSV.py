import csv
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from TimeFreq_plot import Run_spectrogram
import seaborn as sns
from scipy.signal import find_peaks

#col_list = ["CSC10_001_timestamps", "CSC10_001_values","CSC11_015_values","CSC12_000_values","CSC13_013_values","CSC14_002_values"]
col_list = ["CSC10_001_timestamps","CSC1_008_values","CSC2_007_values","CSC3_009_values","CSC4_006_values",
				"CSC5_012_values","CSC6_003_values","CSC7_011_values","CSC8_004_values","CSC9_014_values",
					"CSC10_001_values","CSC11_015_values","CSC12_000_values","CSC13_013_values","CSC14_002_values",
						"CSC15_010_values","CSC16_005_values"]
df = pd.read_csv("DRD 9 8 2017 200ms CSC12 test data.csv", usecols=col_list)
# Time data

Time = df["CSC10_001_timestamps"]
Time = Time.to_numpy()
Time = Time[0:-1:10]

LFP_array=np.zeros(((len(col_list)-1),len(Time)))
for i in range(len(col_list)-1,1,-1):
	print('Electrode = ',col_list[i])
	LFP = df[col_list[i]].to_numpy()
	LFP = LFP[0:-1:10]
	#LFP = LFP-np.mean(LFP)
	LFP_array[i-1,:] = LFP 
print('LFP array test = ',Time[0:10],LFP_array[:,1])#np.shape(LFP_array))
#LFP_array = np.zeros((5,1))
print(np.shape(LFP_array[:,:]))
#for i
#sum()

'''
for i in range(2,len(df.iloc[1,:]),2):
	Row_i = df.iloc[1:6,i]
	print(np.shape(Row_i))
	Row_i = Row_i[:,np.newaxis] #np.reshape(Row_i,(1, Row_i.size))  
	LFP_array = np.append(LFP_array,Row_i,axis = 1)
'''
Electrode_names = list(df)
print(Electrode_names[1][1])
print('Test slice:',LFP_array[:,3])


dt = Time[3]-Time[2]
print(dt)
# Voltage data
V1_1 = df["CSC10_001_values"].to_numpy()
V1_1 = V1_1[0:-1:10]
V1_1 = V1_1-np.mean(V1_1)

Depth = np.arange(0,1400,100)
sigma = 0.3 #S/m
CSD = np.zeros((len(Depth),len(Time)))
for i in range(1,len(Depth)):
	for ii in range(0,len(Time)):
		CSD[i,ii] = sigma*(2*(LFP_array[i,ii])-(LFP_array[i+1,ii])-(LFP_array[i-1,ii]))/((0.000001)**2)

print(np.shape(Time[0:10000]),np.shape(Depth),np.shape(CSD[:,0:10000]))
fig1, (axs1, axs2) = plt.subplots(2, 1,sharex=True,constrained_layout = True)

CSD_plot = axs1.contourf(Time[0:200000],Depth,CSD[:,0:200000])
axs1.set_yticks([0,200,400,600,800,1000,1200,1400])
axs1.set_yticklabels(('1400', '1200', '1000', '800', '600','400', '200', '0'))
cbar = fig1.colorbar(CSD_plot,shrink = 0.9,ax = axs1)
axs2.plot(Time[0:200000],LFP_array[1,0:200000])

plt.show()
