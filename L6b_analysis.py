import csv
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from TimeFreq_plot import Run_spectrogram
'''
with open('DRD 9 8 2017 200ms CSC12 test data.csv', 'r') as csvfile:
	# Extract contents of csv file to a list
	data = csv.reader(csvfile)
	# Convert list to numpy array
	Dat_np = np.array(data)
	print(np.size(Dat_np))
'''
#col_list = [2,32,2]
col_list = ["CSC10_001_timestamps","CSC1_008_values","CSC2_007_values","CSC3_009_values","CSC4_006_values",
				"CSC5_012_values","CSC6_003_values","CSC7_011_values","CSC8_004_values","CSC9_014_values",
					"CSC10_001_values","CSC11_015_values","CSC12_000_values","CSC13_013_values","CSC14_002_values",
						"CSC15_010_values","CSC16_005_values"]
#df = pd.read_csv("DRD 9 8 2017 200ms CSC12 test data.csv", usecols=col_list)

df = pd.read_csv("POM mouse 1.csv", usecols=col_list)



Time = df["CSC10_001_timestamps"]
Time = Time.to_numpy()
V1_1 = df["CSC10_001_values"].to_numpy()
V1_2 = df["CSC11_015_values"].to_numpy()
V1_3 = df["CSC12_000_values"].to_numpy()
V1_4 = df["CSC13_013_values"].to_numpy()
V1_5 = df["CSC14_002_values"].to_numpy()
#V1_1 = V1_1
Time = Time[0:-1:10]
print('dt = ',Time[1]-Time[0])
LFP_array=np.zeros(((len(col_list)-1),len(Time)))
for i in range(len(col_list)-1,1,-1):
	print('Electrode = ',col_list[i])
	LFP = df[col_list[i]].to_numpy()
	LFP = LFP[0:-1:10]
	#LFP = LFP-np.mean(LFP)
	LFP_array[i-1,:] = LFP

for i in range(0,len(LFP_array[:,0])):
	plt.plot(Time[0:50000],LFP_array[i,0:50000]-(1*i))
plt.show()
dt = Time[3]-Time[2]
V1_1 = V1_1[0:-1:10]
V1 = V1_1
V1_1 = V1_1-np.mean(V1_1)
V1_2 = V1_2[0:-1:10]
V1_2 = V1_2-np.mean(V1_2)
V1_3 = V1_3[0:-1:10]
V1_3 = V1_3-np.mean(V1_3)
V1_4 = V1_4[0:-1:10]
V1_4 = V1_4-np.mean(V1_4)
V1_5 = V1_5[0:-1:10]
V1_5 = V1_5-np.mean(V1_5)
print(type(V1_1))
#V1 = np.add(V1_1,V1_2)
#V1 = np.add(V1_1,V1_2)
#V1 = np.add(V1,V1_3)
#V1 = np.add(V1,V1_4)
#V1 = np.add(V1,V1_5)
#V2 = np.subtract(V1,V1_1)

Spect_dat = Run_spectrogram(V1_1,Time)

i_win = np.arange(0,200)
STD_V1 = []
T_win = []
Abs_Grad = []
Abs_Sum = []
srate = 1/dt
nyq = 0.5 * srate

col_name = ["Light pulses"]
ds = pd.read_csv("LightPulses.csv", usecols=col_name)
Pulses = ds["Light pulses"]
Pulses = Pulses.to_numpy()
Pulses = Pulses[0:-1:2]

# Highpass filter
low_cutoff = 0.5
low = low_cutoff / nyq
b, a = signal.butter(5, low, btype='high', analog=False)
V1_filt = signal.filtfilt(b, a, V1)#,axis = 0)

# Create lowpass filter
high_cutoff = 2
high = high_cutoff / nyq
b, a = signal.butter(5, high, btype='low', analog=False)
V1_filt = signal.filtfilt(b, a, V1_filt)#, axis=0)

plt.plot(Time,V1_filt,Time,V1)
plt.show()
####################################################
def Spectrum(V_win):
	LFP_win = V_win - np.mean(V_win)
	#T_win = Time[Window]
	# Windowing if you want
	w = np.hanning(len(V_win))
	# print(type(w))
	LFP_win = w * LFP_win

	# Calculate power spectrum for window
	Fs = srate
	N = len(LFP_win)
	xdft = np.fft.fft(LFP_win)
	# print((N/2)+1)
	xdft = xdft[0:int((N / 2) + 1)]
	psdx = (1 / (Fs * N)) * np.abs(xdft) ** 2
	freq = np.arange(0, (Fs / 2) + Fs / N, Fs / N)
	Pow = psdx
	Pow = np.zeros((501, 1))
	for j in range(0, 200):
		Pow[j] = psdx[j]
	return Pow, freq
###############################################
while (i_win[-1]+200)<=len(Time):
	STD_V1=np.append(STD_V1,np.std(V1_filt[i_win]))
	T_win=np.append(T_win,np.mean(Time[i_win]))
	i_win = np.arange((i_win[0]+25),(i_win[-1]+26))
	Abs_Grad = np.append(Abs_Grad,np.mean(np.abs(np.divide(np.diff(V1[i_win[0:-1:5]]),np.diff(Time[i_win[0:-1:5]]))))) 
	Abs_Sum = np.append(Abs_Sum, np.sum(np.abs(V1[i_win])))

Total_power = np.sum(Spect_dat[0],0)
shift_Total_power = np.append(Total_power[1:],0)
State_change = np.multiply(Total_power,shift_Total_power)
State_change_i = np.where(State_change<0)
State_change_i = State_change_i[0]
State_change_cut = []
for i in range(2,len(State_change_i)-1,2):
	if Spect_dat[1][State_change_i[i]]-Spect_dat[1][State_change_i[i-1]] < 0.5:
		State_change_cut.extend((i-1,i))
State_change_i = np.delete(State_change_i,State_change_cut)
print('State_change_cut ',State_change_cut)

Up_Down = np.zeros(len(STD_V1))
for i in np.arange(0,len(Up_Down)-1):
	if STD_V1[i] >=0.03:
		Up_Down[i] = 1
'''
for i in np.arange(1,len(Up_Down)-2):
	if (Up_Down[i] == 1) and ((Up_Down[i-6] == 1) or (Up_Down[i+1] == 1)):
		print(T_win[i])
		Up_Down[i-6:i] = 1
	else:
		Up_Down[i] = 0

'''
print(Up_Down)
UP_start_i = State_change_i[0,-1,2]
DOWN_start_i = State_change_i[1,-1,2]


#def func(x, a, b, c):
#	return a * np.exp(-b * x) + c

#popt, pcov = curve_fit(func, xdata, ydata)



df = pd.DataFrame({"T_win" : T_win})
df.to_csv("T_win.csv", index=False)

df = pd.DataFrame({"Up_Down" : Up_Down})
df.to_csv("Up_Down .csv", index=False)

Up_start_i_low = np.where(np.diff(Up_Down) == 1)
Down_start_i_low = np.where(np.diff(Up_Down) == -1)
print('Up_Start_i_low shape =',np.shape(Up_start_i_low))
print('Times low = ',T_win[Up_start_i_low])
Up_start_i_low = Up_start_i_low[0]
Down_start_i_low = Down_start_i_low[0]
Up_start_i = np.zeros((1,len(Up_start_i_low)))
Down_start_i = np.zeros((1,len(Down_start_i_low)))
for i in range(0,len(Up_start_i_low)-1):
	Up_start_i_i = np.where(np.logical_and(Time>=T_win[Up_start_i_low[i]]-0.001, Time<=T_win[Up_start_i_low[i]]+0.001))
	Down_start_i_i = np.where(np.logical_and(Time>=T_win[Down_start_i_low[i]]-0.001, Time<=T_win[Down_start_i_low[i]]+0.001))
	Up_start_i[0,i] = Up_start_i_i[0][0]
	Down_start_i[0,i] = Down_start_i_i[0][0]

Up_start_i = Up_start_i.astype(int)
Down_start_i = Down_start_i.astype(int)
print('Up_start = ',Time[Up_start_i])
Up_start_i_low = np.divide(T_win[Up_start_i_low], dt)
Up_start_i_low = Up_start_i_low.astype(int)
#Down_start_i = np.where(np.diff(Up_Down) == -1)
Down_start_i_low = np.divide(T_win[Down_start_i_low],dt)
Down_start_i_low = Down_start_i.astype(int)
print(Time[Up_start_i_low[0]])
print('Up_start_i ',type(Up_start_i[0][0]))
Up_Down_array = TemporaryFile()
T_win_array = TemporaryFile()
np.save('T_win_array',T_win)
np.save('Up_Down_array',Up_Down)
print('time = ',type(Time))
Pulse_triggered = []
for i in range(0, len(Pulses)):
	Pulse_triggered = np.append(Pulse_triggered, np.where(
		np.logical_and(Time[Up_start_i[0]] >= Pulses[i] - 0.5, Time[Up_start_i[0]] <= Pulses[i] + 0.5)))
Pulse_triggered = Pulse_triggered.astype(int)
Pulse_triggered_down = Down_start_i[0][Pulse_triggered]
print('Pulse triggered down: ',Pulse_triggered_down)
Spontan = Up_start_i[0][:]
Spontan = np.delete(Spontan,Pulse_triggered)
Spontan_down = Down_start_i[0][:]
Spontan_down = np.delete(Spontan_down,Pulse_triggered)

Spontan_up_i = Spontan
Spontan_down_i = Spontan_down
Pulse_triggered_up_i = Up_start_i[0][Pulse_triggered]
Pulse_triggered_down_i = Pulse_triggered_down
Pow_S = np.zeros((501,1))
Pow_PT = np.zeros((501,1))
for i in range(0,len(Spontan_up_i)):
	Pow, freq = Spectrum(V1[Spontan_up_i[i]:Spontan_up_i[i]+1000])
	Pow_S = np.append(Pow_S,Pow,axis=1)
for i in range(0,len(Pulse_triggered_up_i)):
	Pow, freq = Spectrum(V1[Pulse_triggered_up_i[i]:Pulse_triggered_up_i[i]+1000])
	Pow_PT = np.append(Pow_PT,Pow,axis=1)
Pow_S_mean = np.mean(Pow_S[:100,1:],axis=1)
Pow_PT_mean = np.mean(Pow_PT[:100,1:],axis=1)
print('Shape power: ',np.shape(Pow_S))
plt.plot(freq[1:101],Pow_S_mean,label='Spontaneous')
plt.plot( freq[1:101],Pow_PT_mean,label='Pulse triggered')
plt.legend(loc='upper right')
plt.show()
print('Spontan down: ',Spontan_down)
print('Pulse triggered up states =', Time[Up_start_i[0][Pulse_triggered]])
print('Spontaneous up states =', Time[Spontan])

Pulse_trig_Duration = np.subtract(Time[Pulse_triggered_down_i],Time[Pulse_triggered_up_i])
Spontan_Duration = np.subtract(Time[Spontan_down_i],Time[Spontan_up_i])
PT_D_x = np.zeros(len(Pulse_trig_Duration))
PT_D_x[:] = 1
S_D_x = np.zeros(len(Spontan_Duration))
S_D_x[:] = 2
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
ax.scatter(PT_D_x,Pulse_trig_Duration,color='blue',marker='o')
ax.scatter(S_D_x,Spontan_Duration,color='green',marker='o')
ax.set_xticks([1, 2])
ax.set_ylabel('Up-state duration (s)')
plt.xlim(0, 3)
ax.set_xticklabels(['Pulse Triggered','Spontaneous'])
plt.show()

fig, (axs1, axs2, axs3, axs4) = plt.subplots(4, 1,sharex=True)

axs1.plot(Time, V1)
#for i in range(0,len(UP_start_i)-1):
#for i in range(0,len(Up_start_i[0])):
#	print(i)
#	axs[0].plot(Time[Up_start_i[0][i]:Down_start_i[0][i]], V1[Up_start_i[0][i]:Down_start_i[0][i]],color = 'green')
for i in range(0,len(Pulse_triggered_up_i)):
	axs1.plot(Time[Pulse_triggered_up_i[i]:Pulse_triggered_down_i[i]], V1[Pulse_triggered_up_i[i]:Pulse_triggered_down_i[i]], color='green')
for i in range(0,len(Spontan_up_i)):
	axs1.plot(Time[Spontan_up_i[i]:Spontan_down_i[i]], V1[Spontan_up_i[i]:Spontan_down_i[i]], color='red')
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



print('Total power size: ',type(Total_power),type(shift_Total_power))

#axs[2].scatter(STD_V1,Abs_Sum,marker = 'o')
'''
fig, axs = plt.subplots(4, 1,sharex=True)

axs[0].plot(Time, V1)
#for i in range(0,len(Up_start_i[0])):
#	print(i)
#	axs[0].plot(Time[Up_start_i[0][i]:Down_start_i[0][i]], V1[Up_start_i[0][i]:Down_start_i[0][i]],color = 'green')
for i in range(0,len(Pulse_triggered_up_i)):
	axs[0].plot(Time[Pulse_triggered_up_i[i]:Pulse_triggered_down_i[i]], V1[Pulse_triggered_up_i[i]:Pulse_triggered_down_i[i]], color='green')
for i in range(0,len(Spontan_up_i)):
	axs[0].plot(Time[Spontan_up_i[i]:Spontan_down_i[i]], V1[Spontan_up_i[i]:Spontan_down_i[i]], color='red')
for i in range(0,len(Pulses)):
	axs[1].plot([Pulses[i], Pulses[i]+0.2],[1, 1],color = 'black',linewidth = 12)
ax[0].set(ylabel = 'LFP (V)')
#axs[1].plot(T_win[:-1],np.diff(Up_Down))
axs[2].plot(T_win,STD_V1)
axs[3].plot(T_win,Up_Down)
#axs[2].scatter(STD_V1,Abs_Sum,marker = 'o')
'''
plt.show()

plt.plot(Spect_dat[1],State_change)
plt.show()

plt.scatter(STD_V1,Abs_Grad,marker = '.')
plt.show()

print('time step = ',dt)
'''
num_frex = 45
range_cycles = [4  ,14]
min_freq = 0.5#0.1
max_freq = 50#45
#frex = np.logspace(np.log10(min_freq),np.log10(max_freq),num = num_frex)

frex = np.linspace(min_freq,max_freq,num = num_frex)

t_wav  = np.arange(-2,(2-(1/srate)),(1/srate))
nCycs = np.logspace(np.log10(range_cycles[0]),np.log10(range_cycles[-1]),num = num_frex)

#s = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex) ./ (2*pi*frex);

half_wave = (len(t_wav)-1)/2;

def spectrogram(y,t_win):
	

	# FFT parameters
	nKern = len(t_wav)
	nData = len(y)
	nConv = nKern+nData-1
	# Convert data to frequency domain
	dataX   = np.fft.fft( y ,nConv )
	tf = np.zeros((num_frex,len(y)-1)) #np.zeros((num_frex,len(t_win)-1))
	#tf = np.zeros((num_frex,nConv-1))

	# Loope through each frequency.
	for fi in range(0,num_frex):
		s = nCycs[fi]/(2*np.pi*frex[fi]);
		# Wavelet function	
		wavelet = np.exp(2*complex(0,1)*np.pi*frex[fi]*t_wav) * np.exp(-t_wav**2/(2*s**2))
		# Wavelet function in the frequency domain
		waveletX = np.fft.fft(wavelet,nConv);
		waveR = [wr.real for wr in wavelet]
		waveI = [wi.imag for wi in wavelet]
		#plt.plot(t_wav,waveR)
		#plt.show()

		# Multiply the fourier transform of the wavelet by the fourier transform of the data
		# then compute the inverse fourier transform to convert back to the time domain.
		As = np.fft.ifft(np.multiply(waveletX,dataX),nConv)
		As = As[int(half_wave)+1:-int(half_wave)]
		#print(np.shape(tf[fi,:]))
		#print(np.shape(As))
		tf[fi,:] = abs(As)**2

	# for i in range(1,num_frex):
   	#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
	#	tf[i,:] = 100 * np.divide(np.subtract(tf[i,:],np.mean(tf[i,:])), np.mean(tf[i,:]))
	Spect_Out = [tf, frex, t_win[0:-1]]
	return Spect_Out
#df = pd.read_csv('DRD 9 8 2017 200ms CSC12 test data.csv', index_col=1)
#df.plot(x=df.index, y=df.columns)
'''
'''
with open("DRD 9 8 2017 200ms CSC12 test data.csv", 'r') as file:
	reader = list(csv.reader(file))
	Data = np.array(reader)
	plt.plot(reader[:,1],reader[:,2])
	print(np.shape(Data))


	#for row in reader:
	#	print(row[1])
'''