import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi

# Step 1 & 2: Generate a noisy signal (5 Hz sine + 50 Hz sine + noise)
fs = 500  # Sampling frequency (Hz)
t = np.linspace(0, 1, fs, endpoint=False)

# Original clean signal components
low_freq = 5   # Hz
high_freq = 50 # Hz

signal = np.sin(2 * np.pi * low_freq * t) + 0.5 * np.sin(2 * np.pi * high_freq * t)
noise = 0.3 * np.random.randn(len(t))
noisy_signal = signal + noise

# Step 3: Design Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Step 4: Filtering function
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y

# Parameters for filtering
cutoff_freq = 10  # Hz (cutoff frequency)
order = 6

# Step 5: Filter the entire signal
filtered_signal = butter_lowpass_filter(noisy_signal, cutoff_freq, fs, order)

# Step 6: Real-time filtering simulation (sample-by-sample)
b, a = butter_lowpass(cutoff_freq, fs, order)
zi = lfilter_zi(b, a) * noisy_signal[0]  # Initialize filter state with first sample

filtered_real_time = []
for sample in noisy_signal:
    filtered_sample, zi = lfilter(b, a, [sample], zi=zi)
    filtered_real_time.append(filtered_sample[0])

# Step 5: Plot results
plt.figure(figsize=(12, 7))
plt.plot(t, signal, label='Original Signal (5 Hz)', linewidth=2)
plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.5)
plt.plot(t, filtered_signal, label='Filtered Signal (Batch)', linewidth=2)
plt.plot(t, filtered_real_time, label='Filtered Signal (Real-Time)', linestyle='--')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.title('Digital Low-Pass Filter (Butterworth) - Batch vs Real-Time')
plt.legend()
plt.grid(True)
plt.show()



