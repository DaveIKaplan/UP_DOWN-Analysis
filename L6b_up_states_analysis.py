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
col_list = ["CSC10_001_timestamps","CSC1_008_values","CSC2_007_values","CSC3_009_values","CSC4_006_values",
				"CSC5_012_values","CSC6_003_values","CSC7_011_values","CSC8_004_values","CSC9_014_values",
					"CSC10_001_values","CSC11_015_values","CSC12_000_values","CSC13_013_values","CSC14_002_values",
						"CSC15_010_values","CSC16_005_values"]
df = pd.read_csv("DRD 9 8 2017 200ms CSC12 test data.csv", usecols=col_list)
# df = pd.read_csv("POM mouse 1.csv", usecols=col_list)
# Time data
Time = df["CSC10_001_timestamps"]
Time = Time.to_numpy()
Time = Time[0:-1:50]
dt = Time[3]-Time[2] #
print('dt = ',dt)

LFP_array=np.zeros(((len(col_list)-1),len(Time)))
for i in range(len(col_list)-1,0,-1):
	print('Electrode = ',col_list[i])
	LFP = df[col_list[i]].to_numpy()
	LFP = LFP[0:-1:50]
	#LFP = LFP-np.mean(LFP)
	LFP_array[i-1,:] = LFP

for i in range(0,len(LFP_array[:,0])):
	plt.plot(Time,LFP_array[i,:]-(1*i))
plt.show()
# Voltage data for electrode 10
V1_1 = df["CSC10_001_values"].to_numpy()
V1_1 = V1_1[0:-1:50]
V1_1 = V1_1-np.mean(V1_1)
# Light pulse data
#col_name = ["Light pulses"]
#col_name = ["EvMarker"]
#ds = pd.read_csv("LightPulses.csv",sep = " ", usecols=col_name)
#ds = pd.read_csv("POM mouse 1 event times (b).txt",sep = " ", usecols=col_name)
#Pulses = ds["EvMarker"]
#Pulses = ds["Light pulses"]

col_name = ["Light pulses"]
ds = pd.read_csv("LightPulses.csv", usecols=col_name)
Pulses = ds["Light pulses"]

print(Pulses)
Pulses = Pulses.to_numpy()
Pulses = Pulses[0:-1:2]
i_win = np.arange(0,200)
T_win = []


UP_Time = np.arange(0,2-(1*dt),dt)
Spect_dat = Run_spectrogram(V1_1,Time)


# Power spectrum parameters
srate = 1/dt
nyq = 0.5 * srate
freq = np.arange(0, (srate / 2) + srate / len(UP_Time), srate / len(UP_Time))

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
	Pow = np.zeros((501, 1))
	for j in range(0, 200):
		Pow[j] = psdx[j]
	return Pow, freq

# Find up states by finding peak power in 1 - 10Hz range
Total_power = np.sum(Spect_dat[0],0) # Sum the power at every timestep between 0 and 10Hz
shift_Total_power = np.append(Total_power[1:],0) # Create another array that is Total_power, but offset by 1 timestep
State_change = np.multiply(Total_power,shift_Total_power) # To find where the total power crosses 0 multiply by the
# offset array, where the values cross the zero line the diff
# will be negative (- * + = -). Every where else it is positive.
State_change_i = np.where(State_change<0)  # Find the corresponding timepoints (i.e. where total power = 0)
State_change_i = State_change_i[0] # Takes the first element of the list returned using np.where - gives the array of values

# Remove all Up-Down transitions that are less than 0.5s
State_change_cut = [] # Declare the list of state transitions that will be cut
for i in range(2,len(State_change_i)-1,2): # Loop through the Up transitions starting from the second Up state
	# Find the Up states that are less than 0.5s from the end of the preceding Up state
	# Spect_dat[1] = time signal from the sepctrogram output
	if Spect_dat[1][State_change_i[i]]-Spect_dat[1][State_change_i[i-1]] < 0.5: # If the time difference between the start
																				# of an upstate and the end of the preceding Up state
																				# is less than 0.5s...
		State_change_cut.extend((i-1,i))	# .. append it to the State_change_cut array.
State_change_i = np.delete(State_change_i,State_change_cut) # Delete all state changes that are < 0.5s

# Up and Down state indexes in downsampled data todo consolidate all UP DOW state arrays
UP_start_i = State_change_i[2::2] # Identify UP transitions
print('UP states = ',UP_start_i)
DOWN_start_i = State_change_i[3::2] # Identify DOWN transitions



# Create an array of the times when a pulse is occurring
Pulse_times_array = []
for i in range(0,len(Pulses)): # Iterate through the pulses
	Pulse_times = np.arange(Pulses[i],Pulses[i]+0.3,0.005) # For each pulse make an array of the time window the array covers
	Pulse_times_array = np.append(Pulse_times_array,Pulse_times) # Append this for each new pulse
# Iterate through the UP and DOWN start times and find UP states that coincide with a pulse
Pulse_times_array
UP_start_i
DOWN_start_i
Pulse_ON = np.ones(len(Pulse_times_array))
#Fig, axs = plt.subplots(3,1,sharex=True)
#axs[0].plot(Time,V1_1)
#axs[0].scatter(Time[UP_start_i],V1_1[UP_start_i],marker = 'o',color = 'green')
#axs[0].scatter(Time[DOWN_start_i],V1_1[DOWN_start_i],marker = 'o',color = 'red')
#plt.ylabel('LFP')
#axs[1].scatter(Pulse_times_array,Pulse_ON)
#plt.ylabel('Pulse')
#plt.show()
print('Pulse Times = ',Pulse_times_array)
Pulse_triggered_array = [[1,0]]
for i in range(len(UP_start_i)-1): # todo check to make sure indexing is consistent across the next line
	# Find where pulse array coincides with the time window 50ms before every up state (i.e. triggering an UP state)
	Test_pulse_triggered = np.where(np.logical_and(Pulse_times_array>=Time[UP_start_i[i]]-0.05,Pulse_times_array<=Time[UP_start_i[i]]))
	if len(Test_pulse_triggered[0])>0:
		# Add the UP and DOWN states start times to the pulse-triggered array
		Pulse_triggered_array = np.concatenate((Pulse_triggered_array,[[UP_start_i[i], DOWN_start_i[i]]]),axis = 0)
Pulse_associated_array = [[1,0]]
for j in range(0,len(UP_start_i)-1):
	# Find where pulse array coincides with an UP state
	Test_pulse_triggered = np.where(np.logical_and(Pulse_times_array>=Time[UP_start_i[j]],Pulse_times_array<=Time[DOWN_start_i[j]]))
	if len(Test_pulse_triggered[0])>0:
		#input("We have UP coincedent pulses!")
		Pulse_associated_array = np.concatenate((Pulse_associated_array,[[UP_start_i[j], DOWN_start_i[j]]]),axis = 0)
	#else:
		#input("NO UP coincident pulses :(")
Pulse_triggered_UP = Pulse_triggered_array[1:,0]
Pulse_triggered_DOWN = Pulse_triggered_array[1:,1]
Pulse_associated_up = Pulse_associated_array[1:,0]
Pulse_associated_down = Pulse_associated_array[1:,1]
# Identify any UP states in the Pulse_associated_array that are triggered by a pulse and exclude those from the pulse associated
# Up and Down start times
Pulse_associated_UP = [k for k, x in enumerate(np.isin(Pulse_associated_up,Pulse_triggered_UP)) if not x] # todo: figure out what happened to the pulse associated UP states
Pulse_associated_UP = Pulse_associated_up[Pulse_associated_UP]
Pulse_associated_DOWN = [l for l, x in enumerate(np.isin(Pulse_associated_down,Pulse_triggered_DOWN)) if not x]
Pulse_associated_DOWN = Pulse_associated_down[Pulse_associated_DOWN]

# Build an  array of all UP states that are triggered or associated with a pulse
Pulse_coincident_UP = np.append(Pulse_triggered_UP,Pulse_associated_UP) # Array of all UP states that coincide with a pulse
print('Up start times that coincide with a pulse: ',Pulse_coincident_UP)
Pulse_coincident_DOWN = [[Pulse_triggered_DOWN],[Pulse_associated_DOWN]]
Spontaneous_UP_x = [i for i, x in enumerate(np.isin(UP_start_i,Pulse_triggered_UP)) if not x]
Spontaneous_UP = UP_start_i[Spontaneous_UP_x]
Spontaneous_UP_x = [i for i, x in enumerate(np.isin(Spontaneous_UP,Pulse_associated_UP)) if not x]
Spontaneous_UP = Spontaneous_UP[Spontaneous_UP_x]
Spontaneous_DOWN_x = [i for i, x in enumerate(np.isin(DOWN_start_i,Pulse_triggered_DOWN)) if not x]
Spontaneous_DOWN = DOWN_start_i[Spontaneous_DOWN_x]
Spontaneous_DOWN_x = [i for i, x in enumerate(np.isin(Spontaneous_DOWN,Pulse_associated_DOWN)) if not x]
Spontaneous_DOWN = Spontaneous_DOWN[Spontaneous_DOWN_x]

print('UP states:',Spontaneous_UP,Pulse_triggered_UP,Pulse_associated_UP)
print('Spon UP =',Spontaneous_UP)
print('Spon DOWN =',Spontaneous_DOWN)
Spontaneous_UP = Spontaneous_UP[:-1]
Duration_Pulse_Triggered = np.subtract(Time[Pulse_triggered_DOWN],Time[Pulse_triggered_UP])
Duration_Spontaneous = np.subtract(Time[Spontaneous_DOWN],Time[Spontaneous_UP])
Duration_Pulse_Associated = np.subtract(Time[Pulse_associated_DOWN],Time[Pulse_associated_UP])
print('Test that UP and DOWN states are paired = ',len(UP_start_i),len(DOWN_start_i))
Pulse_Triggered_Labels = ['Pulse Triggered']*len(Duration_Pulse_Triggered)
Spontaneous_Labels = ['Spontaneous']*len(Duration_Spontaneous)
Pulse_associated_Labels = ['Pulse Assoc.']*len(Duration_Pulse_Associated)
Labels = np.append(Pulse_Triggered_Labels, Spontaneous_Labels)
Labels = np.append(Labels, Pulse_associated_Labels)
Durations = np.append(Duration_Pulse_Triggered, Duration_Spontaneous)
Durations = np.append(Durations,Duration_Pulse_Associated)
print('Lengths',np.shape(Labels),np.shape(Durations))
#sns.pointplot(Labels,Durations)
sns.stripplot(Labels, Durations,x="UP States", y="Duration (s)",  dodge=True, alpha=.25, zorder=1)
plt.ylabel('UP state duration')
plt.show()

print('Pulsed UP:', Pulse_triggered_UP)
print('Spontaneous UP:', Spontaneous_UP)
print('Pulse associated UP:', Pulse_coincident_UP)


Pulse_times_array
UP_start_i
DOWN_start_i

# Run each UP state through spectrogram and build an array
# First for pulse triggered, then spontaneous
Spont_UP_Power = np.zeros((len(Spontaneous_UP),))

# Run a for loop that iterates through every spontaneous UP state and writes it to an array
# So each row of the array is a different UP state and the same length, padded with zeros
fig1, (axs1, axs2) = plt.subplots(2, 1,sharex=True)


t_array = np.arange(-0.75,2-dt,dt)

# Create lowpass filter
high_cutoff = 10
high = high_cutoff / nyq
b_lp, a_lp = signal.butter(5, high, btype='low', analog=False)

# Highpass filter
low_cutoff = 2
low = low_cutoff / nyq
b_hp, a_hp = signal.butter(5, low, btype='high', analog=False)

##### MAKE ARRAY OF Spontaneous UP states
Spon_UP_array = np.zeros((len(Spontaneous_UP),int(2.75/dt)))
for i_Spon in range(0,len(Spontaneous_UP)):
	Spon_UP_array[i_Spon] = V1_1[Spontaneous_UP[i_Spon]-int(0.75/dt):Spontaneous_UP[i_Spon]+int(2/dt)]
##### MAKE PEAK ALLIGNED ARRAY OF Spontaneous UP states
Spon_UP_peak_alligned_array = np.zeros((len(Spontaneous_UP),int(2/dt)))
Spon_Peaks = []
for i_Spon in range(len(Spontaneous_UP)): # todo low pass filter data
	print('i spon = ',i_Spon)
	V_filt = signal.filtfilt(b_lp, a_lp, Spon_UP_array[i_Spon])  # , axis=0)
	V_filt = signal.filtfilt(b_hp, a_hp, V_filt)
	peaks, _ = find_peaks(V_filt[int(0.25/dt):int(1.25/dt)], height = np.round(np.std(V_filt),3), distance=150)
	peaks += int(0.25/dt)
	#peaks = peaks[np.argmax(V_filt[peaks])] if len(peaks)>1 else None
	print('i spon next = ',i_Spon)
	Spon_Peaks = np.append(Spon_Peaks,(peaks[0]-int(0.75/dt)+Spontaneous_UP[i_Spon]))
	#plt.plot(t_array, Spon_UP_array[i_Spon],label = 'Raw',zorder=1)
	#plt.plot(t_array, V_filt,label = 'Filtered',zorder=2)
	#plt.legend()
	#plt.scatter(t_array[peaks], V_filt[peaks],marker = 'o',s=80,facecolors='none', edgecolors='r',zorder=3)
	#Plot_Label = "Up state: {}, Std = {}".format(i_Spon,np.round(np.std(V_filt),3))
	#plt.title(Plot_Label)
	#plt.show()
	Spon_UP_peak_alligned_array[i_Spon] = Spon_UP_array[i_Spon][peaks[0]-int(0.5/dt):peaks[0]+int(1.5/dt)]
print('Spon Peaks = ',Spon_Peaks)
Spon_Peaks = Spon_Peaks.astype(int)

plt.plot(Time,V1_1,zorder = 1)
plt.scatter(Time[Spon_Peaks],V1_1[Spon_Peaks],color = 'red', zorder = 2)
plt.show()

for i_Spon in range(0,len(Spontaneous_UP)):
	axs1.plot(UP_Time,Spon_UP_peak_alligned_array[i_Spon],color = 'blue',linewidth = 0.2, alpha = 0.2)
	#plt.plot(UP_Time[peaks,Spon_UP_array[i_Spon][peaks],'o',color = 'red',linewidth = 0.2, alpha = 0.5)
	#plt.show()
#input('Size of peak alligned array and time step = {}, {}'.format(len(Spon_UP_peak_alligned_array[0,:]),(UP_Time[1]-UP_Time[0])))
axs1.plot(UP_Time,np.mean(Spon_UP_peak_alligned_array,0),color = 'blue')

# Run a for loop that iterates through every spontaneous UP state and writes it to an array
# So each row of the array is a different UP state and the same length, padded with zeros
##### MAKE ARRAY OF Pulse triggered UP states
Pulsed_UP_array = np.zeros((len(Pulse_triggered_UP),int(2.75/dt)))
for i_Pulsed in range(0,len(Pulse_triggered_UP)):
	Pulsed_UP_array[i_Pulsed] = V1_1[Pulse_triggered_UP[i_Pulsed]-int(0.75/dt):Pulse_triggered_UP[i_Pulsed]+int(2/dt)]
Pulsed_UP_peak_alligned_array = np.zeros((len(Pulse_triggered_UP),int(2/dt)))
##### MAKE PEAK ALLIGNED ARRAY OF Pulse triggered UP states
Pulsed_Peaks = []
for i_Pulsed in range(0,len(Pulse_triggered_UP)):
	print(i_Pulsed)
	V_filt = signal.filtfilt(b_lp, a_lp, Pulsed_UP_array[i_Pulsed])  # , axis=0)
	V_filt = signal.filtfilt(b_hp, a_hp, V_filt)
	peaks, _ = find_peaks(V_filt[0:int(1.25 / dt)], height=np.round(np.std(V_filt), 3), distance=150)
	#plt.plot(t_array, Pulsed_UP_array[i_Pulsed], label='Raw', zorder=1)
	#plt.plot(t_array, V_filt, label='Filtered', zorder=2)
	#plt.legend()
	#plt.scatter(t_array[peaks], V_filt[peaks], marker='o', s=80, facecolors='none', edgecolors='r', zorder=3)
	#Plot_Label = "Up state: {}, Std = {}".format(i_Pulsed, np.round(np.std(V_filt), 3))
	#plt.title(Plot_Label)
	#plt.show()
	Pulsed_Peaks = np.append(Pulsed_Peaks, (peaks[0] - int(0.75 / dt) + Pulse_triggered_UP[i_Pulsed]))
	Pulsed_UP_peak_alligned_array[i_Pulsed] = Pulsed_UP_array[i_Pulsed][peaks[0]-int(0.5/dt):peaks[0]+int(1.5/dt)]
Pulsed_Peaks = Pulsed_Peaks.astype(int)
for i_Pulsed in range(0,len(Pulse_triggered_UP)):
	axs2.plot(UP_Time,Pulsed_UP_peak_alligned_array[i_Pulsed],color = 'red',linewidth = 0.2, alpha = 0.2)
#	plt.show()
print('Peaks sizes',len(Pulsed_Peaks),len(Spon_Peaks))
axs2.plot(UP_Time,np.mean(Pulsed_UP_peak_alligned_array,0),color = 'red')
plt.show()
print('Pulsed Peaks = ',Pulsed_Peaks)
#################################### HERE START
##### MAKE ARRAY OF Pulse triggered UP states
Pulsed_coincident_UP_array = np.zeros((len(Pulse_coincident_UP),int(2.75/dt)))
for i_Puls_coinc in range(0,len(Pulse_coincident_UP)):
	Pulsed_coincident_UP_array[i_Puls_coinc] = V1_1[Pulse_coincident_UP[i_Puls_coinc]-int(0.75/dt):Pulse_coincident_UP[i_Puls_coinc]+int(2/dt)]
Pulsed_coincident_UP_peak_alligned_array = np.zeros((len(Pulse_coincident_UP),int(2/dt)))
##### MAKE PEAK ALLIGNED ARRAY OF Pulse triggered UP states
for i_Puls_coinc in range(0,len(Pulse_coincident_UP)):
	print(i_Puls_coinc)
	V_filt = signal.filtfilt(b_lp, a_lp, Pulsed_coincident_UP_array[i_Puls_coinc])  # , axis=0)
	V_filt = signal.filtfilt(b_hp, a_hp, V_filt)
	peaks, _ = find_peaks(V_filt[0:int(1.25 / dt)], height=np.round(np.std(V_filt), 3), distance=150)
	#plt.plot(t_array, Pulsed_coincident_UP_array[i_Puls_coinc], label='Raw', zorder=1)
	#plt.plot(t_array, V_filt, label='Filtered', zorder=2)
	#plt.legend()
	#plt.scatter(t_array[peaks], V_filt[peaks], marker='o', s=80, facecolors='none', edgecolors='r', zorder=3)
	#Plot_Label = "Up state: {}, Std = {}".format(i_Puls_coinc, np.round(np.std(V_filt), 3))
	#plt.title(Plot_Label)
	#plt.show()

	Pulsed_coincident_UP_peak_alligned_array[i_Puls_coinc] = Pulsed_coincident_UP_array[i_Puls_coinc][peaks[0]-int(0.5/dt):peaks[0]+int(1.5/dt)]

for i_Puls_coinc in range(0,len(Pulse_coincident_UP)):
	axs2.plot(UP_Time,Pulsed_coincident_UP_peak_alligned_array[i_Puls_coinc],color = 'red',linewidth = 0.2, alpha = 0.2)
#	plt.show()

axs2.plot(UP_Time,np.mean(Pulsed_coincident_UP_peak_alligned_array,0),color = 'red')
plt.show()
#################################### HERE END
##### X CORRELATION ########### todo cross correlations
'''
For the peak alligned UP state array, iterate over each line and perform a cross correlation
'''
print(Spon_UP_peak_alligned_array[:,0])
t_short = np.arange(0.25,1.5,0.005)
Spon_UP_correlations = np.zeros((len(Spon_UP_peak_alligned_array[:,0]),len(Spon_UP_peak_alligned_array[:,0])))
Pulse_triggered_UP_correlations = np.zeros((len(Spon_UP_peak_alligned_array[:,0]),len(Pulsed_UP_peak_alligned_array[:,0])))
Pulse_coinc_UP_correlations = np.zeros((len(Spon_UP_peak_alligned_array[:,0]),len(Pulsed_coincident_UP_peak_alligned_array[:,0])))

for i in range(len(Spon_UP_peak_alligned_array[:,0])):
	for j in range(len(Spon_UP_peak_alligned_array[:,0])):
		J_xcorr = signal.correlate(Spon_UP_peak_alligned_array[i,50:300], Spon_UP_peak_alligned_array[j,50:300], mode='same', method='direct')
		#fig_xcorr, (axs1, axs2) = plt.subplots(2, 1, sharex=True)
		#axs1.plot(t_short,Spon_UP_peak_alligned_array[i,50:300])
		#axs2.plot(t_short,Spon_UP_peak_alligned_array[j,50:300])
		#plt.show()
		r, p = stats.pearsonr(Spon_UP_peak_alligned_array[i,50:300], Spon_UP_peak_alligned_array[j,50:300])
		#plt.plot(J_xcorr)
		Spon_UP_correlations[i,j] = np.abs(r)
		#Label = 'Pearson r is {}'.format(r)
		#plt.title(Label)
		#plt.show()
	for j in range(len(Pulsed_coincident_UP_peak_alligned_array[:,0])):
		J_xcorr = signal.correlate(Spon_UP_peak_alligned_array[i,50:300], Pulsed_coincident_UP_peak_alligned_array[j,50:300], mode='same', method='direct')
		#fig_xcorr, (axs1, axs2) = plt.subplots(2, 1, sharex=True)
		#axs1.plot(t_short,Spon_UP_peak_alligned_array[i,50:300])
		#axs2.plot(t_short,Spon_UP_peak_alligned_array[j,50:300])
		#plt.show()
		r, p = stats.pearsonr(Spon_UP_peak_alligned_array[i,50:300], Pulsed_coincident_UP_peak_alligned_array[j,50:300])
		Pulse_coinc_UP_correlations[i,j] = np.abs(r)
		#plt.plot(J_xcorr)
		#Label = 'Pearson r is {}'.format(r)
		#plt.title(Label)
		#plt.show()
	for j in range(len(Pulsed_UP_peak_alligned_array[:,0])):
		J_xcorr = signal.correlate(Spon_UP_peak_alligned_array[i,50:300], Pulsed_UP_peak_alligned_array[j,50:300], mode='same', method='direct')
		#fig_xcorr, (axs1, axs2) = plt.subplots(2, 1, sharex=True)
		#axs1.plot(t_short,Spon_UP_peak_alligned_array[i,50:300])
		#axs2.plot(t_short,Spon_UP_peak_alligned_array[j,50:300])
		#plt.show()
		r, p = stats.pearsonr(Spon_UP_peak_alligned_array[i,50:300], Pulsed_UP_peak_alligned_array[j,50:300])
		Pulse_triggered_UP_correlations[i,j] = np.abs(r)
		#plt.plot(J_xcorr)
		#Label = 'Pearson r is {}'.format(r)
		#plt.title(Label)
		#plt.show()
print('asdfasd dfas jk')
UP_correlations = np.c_[Spon_UP_correlations,Pulse_coinc_UP_correlations,Pulse_triggered_UP_correlations]
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
pos1 = ax1.imshow(Spon_UP_correlations, cmap='Blues', interpolation='none')
ax1.set_title('Spontaneous vs. Spontaneous')
fig.colorbar(pos1, ax=ax1)
pos2 = ax2.imshow(Pulse_coinc_UP_correlations, cmap='Reds', interpolation='none')
fig.colorbar(pos2, ax=ax2)
ax2.set_title('Spontaneous vs. Pulse Coincident')
pos3 = ax3.imshow(Pulse_triggered_UP_correlations, cmap='Greens', interpolation='none')
fig.colorbar(pos3, ax=ax3)
ax3.set_title('Spontaneous vs. Pulse Triggered')

#ax1.title('Spon. vs Spon. correlations')
plt.show()
#plt.imshow(Spon_UP_correlations)
Spon_UP_correlations_vec = []
for i in range(len(Spon_UP_correlations)):
	for j in range(1,len(Spon_UP_correlations)-i):
		Spon_UP_correlations_vec = np.append(Spon_UP_correlations_vec,Spon_UP_correlations[i,i+j])

Spon_UP_correlations_vec = np.ravel(Spon_UP_correlations_vec)
Pulse_triggered_UP_correlations_vec = np.ravel(Pulse_triggered_UP_correlations)
Pulse_coinc_UP_correlations_vec = np.ravel(Pulse_coinc_UP_correlations)
Label_Spon = ['Spontaneous']*len(Spon_UP_correlations_vec)
Label_Trig = ['Pulse triggered']*len(Pulse_triggered_UP_correlations_vec)
Label_Coinc = ['Pulse Coincident']*len(Pulse_coinc_UP_correlations_vec)
Labels_Corr = np.append(Label_Spon,Label_Trig)
Labels_Corr = np.append(Labels_Corr,Label_Coinc)
Data_Corr = np.append(Spon_UP_correlations_vec,Pulse_triggered_UP_correlations_vec)
Data_Corr = np.append(Data_Corr,Pulse_coinc_UP_correlations_vec)
print('Test correlation seaborn data size ', len(Labels_Corr),len(Data_Corr))
#sns.stripplot(Labels_Corr, Data_Corr,x="UP States", y="Pearson correlation (r)", size=4, color=".3", linewidth=0)#  dodge=True, alpha=.25, zorder=1)
#sns.boxplot(Labels_Corr, Data_Corr,x="UP States", y="Pearson correlation (r)", whis=[0, 100], width=.6, palette="vlag")
sns.set_theme(style="ticks", palette="pastel")
sns.boxplot(Labels_Corr, Data_Corr,x="UP States",y="Pearson correlation (r)",width=.5)
plt.show()
#signal.correlate(in1, in2, mode='full', method='auto')
############ CSD Average for peak triggered events ########
def CSD_mean(Peaks_vector):
	CSD_array = np.zeros((len(Peaks_vector),len(LFP_array[:,0])-2,208))
	for i in range(len(Peaks_vector)):
		print('Peak vectors length',len(Peaks_vector))
		CSD_out = CSD_calc(LFP_array[:,(Peaks_vector[i]-int(0.5/dt)):(Peaks_vector[i]+int(0.5/dt))],dt)
		CSD_array[i,:,:]=CSD_out
		CSD_array_mean = np.mean(CSD_array,axis=0)
	return CSD_array_mean
Electrode_no = np.arange(0,14,1)
Electrode_depth = np.arange(1400,0,-200)
UP_time_window = np.arange(-0.5,0.5-dt,dt)
Spon_CSD_mean = CSD_mean(Spon_Peaks)
Pulsed_CSD_mean = CSD_mean(Pulsed_Peaks)
print('Check CSD means',Spon_CSD_mean[10,10],Pulsed_CSD_mean[10,10])
fig, (axs1, axs2) = plt.subplots(1,2)
pos1 = axs1.contourf(UP_time_window, Electrode_no,  Spon_CSD_mean)
axs1.set(title = 'Spontaneous')
axs1.set_yticklabels(Electrode_depth)
pos2 = axs2.contourf(Pulsed_CSD_mean)
axs2.set(title = 'Pulse triggered')
fig.colorbar(pos2, ax=axs2)
'''
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
pos1 = ax1.imshow(Spon_UP_correlations, cmap='Blues', interpolation='none')
ax1.set_title('Spontaneous vs. Spontaneous')
fig.colorbar(pos1, ax=ax1)
'''
plt.show()
# Power spectra for different UP states
Freq = np.arange(0,501)
print('Freq =',len(freq))
Spon_UP_spectrum_array = np.zeros((len(Spon_UP_peak_alligned_array),501))
for Spon_spect_i in range(0,len(Spontaneous_UP)):
	Pow, freq = Spectrum(Spon_UP_peak_alligned_array[Spon_spect_i])
	Pow = np.squeeze(Pow)
	print('Power = ',np.shape(Spon_UP_spectrum_array))
	Spon_UP_spectrum_array[Spon_spect_i] = Pow
	plt.plot(Freq,Pow,color = 'blue',linewidth = 0.2, alpha = 0.2)
plt.plot(Freq,np.mean(Spon_UP_spectrum_array,0),color='blue')
Pulsed_UP_spectrum_array = np.zeros((len(Pulsed_UP_peak_alligned_array),501))
for Pulsed_spect_i in range(0,len(Pulse_triggered_UP)):
	Pow, freq = Spectrum(Pulsed_UP_peak_alligned_array[Pulsed_spect_i])
	Pow = np.squeeze(Pow)
	print('Power = ',np.shape(Pulsed_UP_spectrum_array))
	Pulsed_UP_spectrum_array[Pulsed_spect_i] = Pow
	plt.plot(Freq,Pow,color = 'red',linewidth = 0.2, alpha = 0.2)
plt.plot(Freq,np.mean(Pulsed_UP_spectrum_array,0),color='red')
plt.plot()
print('Test array generation:',Spon_UP_array)
print('Spon down = ',Spontaneous_DOWN)

fig, (axs1, axs2, axs3, axs4) = plt.subplots(4, 1,sharex=True)

axs1.plot(Time, V1_1)
axs1.scatter(Time[UP_start_i],V1_1[UP_start_i],marker = 'o',color = 'red')
axs1.scatter(Time[DOWN_start_i],V1_1[DOWN_start_i],marker = 'o',color = 'blue')
#for i in range(0,len(UP_start_i)-1):
for i in range(0,len(Spontaneous_UP)):
#	print(i)
	axs1.plot(Time[Spontaneous_UP[i]:Spontaneous_DOWN[i]], V1_1[Spontaneous_UP[i]:Spontaneous_DOWN[i]],color = 'black')
for i in range(0,len(Pulse_triggered_UP)):
#	print(i)
	axs1.plot(Time[Pulse_triggered_UP[i]:Pulse_triggered_DOWN[i]], V1_1[Pulse_triggered_UP[i]:Pulse_triggered_DOWN[i]],color = 'red')
for i in range(0,len(Pulse_associated_UP)):
#	print(i)
	axs1.plot(Time[Pulse_associated_UP[i]:Pulse_associated_DOWN[i]], V1_1[Pulse_associated_UP[i]:Pulse_associated_DOWN[i]],color = 'green',linestyle = 'dotted')
'''
for i in range(0,len(Pulse_triggered_up_i)):
	axs1.plot(Time[Pulse_triggered_up_i[i]:Pulse_triggered_down_i[i]], V1[Pulse_triggered_up_i[i]:Pulse_triggered_down_i[i]], color='green')
for i in range(0,len(Spontan_up_i)):
	axs1.plot(Time[Spontan_up_i[i]:Spontan_down_i[i]], V1[Spontan_up_i[i]:Spontan_down_i[i]], color='red')
'''
for i in range(0,len(Pulses)):
	axs2.plot([Pulses[i], Pulses[i]+0.2],[1, 1],color = 'black',linewidth = 12)


axs1.set(ylabel = 'LFP (V)')
axs2.set(ylabel = 'Light pulses')
axs3.set(ylabel = 'Filtered LFP stdv')
axs4.set(ylabel = 'Up States')
axs4.set(xlabel = 'Time (s)')
#axs[1].plot(T_win[:-1],np.diff(Up_Down))
#axs3.plot(T_win,STD_V1)
axs3.contourf(Spect_dat[1],Spect_dat[2],Spect_dat[0])
#axs4.plot(T_win,Up_Down)
axs4.plot(Spect_dat[1],Total_power)
axs4.scatter(Spect_dat[1][State_change_i],Total_power[State_change_i],marker = '.')
x=np.zeros(len(Pulse_times_array))
plt.scatter(Pulse_times_array,x)
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
# Check that Spontaneous UP and Spontaneous down are correct

print('Pulse triggered Up states:',Pulse_triggered_array[:,0])
plt.show()
df = pd.DataFrame({"T_win" : T_win})
df.to_csv("T_win.csv", index=False)

df = pd.DataFrame({"Up_Down" : Up_Down})
df.to_csv("Up_Down .csv", index=False)


# Highpass filter
low_cutoff = 0.5
low = low_cutoff / nyq
b, a = signal.butter(5, low, btype='high', analog=False)
V1_filt = signal.filtfilt(b, a, V1_1)#,axis = 0)

