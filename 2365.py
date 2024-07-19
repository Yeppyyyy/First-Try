import numpy as np
from scipy.fft import fft
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import kurtogram as kur
df = pd.read_csv('C:/Users/Gjh/Desktop/VibrationSignal/2365.csv')

def envelope(data):
    analytic_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

first_column= df.iloc[:, 0]
xEnv = envelope(list(first_column))

fs = 25.6*10**3
t = np.arange(0,len(first_column))/fs
plt.figure(figsize=(10, 6))
plt.plot(t,first_column, label='Original')
plt.plot(t,xEnv, label='Envelope')
plt.title("Time information(2365)")
plt.xlabel("time(s)")
plt.ylabel("amplitude")
plt.xlim(0.05,0.1)
plt.legend()

w = np.linspace(0,2*np.pi,len(first_column))[1:]
ft = fft(list(xEnv))[1:]
plt.figure(figsize=(10, 6))
plt.plot(w/2/np.pi*fs,np.abs(ft))
plt.title("Frequency information(2365)")
plt.xlabel("f(Hz)")
plt.ylabel("amplitude")
plt.xlim(0,700)
n = 8
fr = 40
d = 7.92
D = 34.55
BPFO = n*fr/2*(1-d/D)
A = [BPFO,BPFO*2,BPFO*3,BPFO*4,BPFO*5]
for a in A:
    plt.axvline(x=a, color='r', linestyle='--')

plt.legend()

plt.figure()
Kwav, Level_w, freq_w, c ,fc,BW= kur.fast_kurtogram(list(first_column),fs)#oo是时间输入
plt.imshow(np.clip(Kwav,0,np.inf),aspect=10)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y



# Bandpass filter design
lowcut = fc - BW/4
highcut = fc + BW/4
b, a = butter_bandpass(lowcut, highcut, fs, order=6)

# Apply bandpass filter
xOuterBpf = butter_bandpass_filter(list(first_column), lowcut, highcut, fs)

# Extract envelope of filtered signal
xEnvOuterBpf = envelope(xOuterBpf)

# Plotting
tOuter = np.linspace(0, len(list(first_column))/fs, len(list(first_column)))
tEnvOuterBpf = np.linspace(0, len(xEnvOuterBpf)/fs, len(xEnvOuterBpf))



plt.figure(figsize=(10, 6))
plt.plot(tOuter, xOuterBpf, label='Original after filtering')
plt.plot(tEnvOuterBpf, xEnvOuterBpf, label='Envelope after filtering')
plt.ylabel('amplitude')
plt.xlabel('Time (s)')
#plt.title('Bandpass Filtered Signal: Outer Race Fault, kurtosis = {:.2f}'.format(kurtOuterBpf))
plt.xlim([0, 0.1])
plt.legend()

w = np.linspace(0,2*np.pi,len(xEnvOuterBpf))[1:]
ft = fft(list(xEnvOuterBpf))[1:]
plt.figure(figsize=(10, 6))
plt.plot(w/2/np.pi*fs,np.abs(ft))
plt.title("Frequency information(2365)")
plt.xlabel("f(Hz)")
plt.ylabel("amplitude")
plt.xlim(0,700)
n = 8
fr = 40
d = 7.92
D = 34.55
BPFO = n*fr/2*(1-d/D)
A = [BPFO,BPFO*2,BPFO*3,BPFO*4,BPFO*5]
for a in A:
    plt.axvline(x=a, color='r', linestyle='--')

plt.legend()


plt.show()