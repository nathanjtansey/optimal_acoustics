import scipy
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import math
import seaborn as sns
from librosa import display
from scipy.fft import fft

def get_default_params():
    window_size = 512
    step = int(window_size/2)
    bins = 20
    return(window_size,step,bins)

### info calculation functions ###

# -- useful intermediate function for waveform visuals of different windows
def stepper(data, rate, window_size = 0, step_size = 0):  # useful intermediate function for waveform visuals of different windows
    
    if window_size == 0:           
        window = rate # sets default window size to the rate
    else:
        window = window_size 
    
    if step_size == 0:
        step = int(window/2) # sets default step size to half of the window size
    else: 
        step = step_size    
    
    
    stepTotal = math.floor(len(data) / step) # rounds up on number of steps
    stepNum = 0
    stepList = np.zeros((stepTotal,window))

    while stepNum*step < len(data) - window: # allows for a truncated window on the very last step to account for the rounding up on the number of steps
        stepList[stepNum,:] = data[stepNum*step: stepNum*step + window]
        stepNum += 1
    tmp_tail = data[0+stepNum*step: len(data)] 
    stepList[stepNum,:len(tmp_tail)] = tmp_tail 
    return(stepList)


def log_time_series_power(stepList):
    max_power = np.max(np.abs(np.ravel(stepList)))
    threshold_mask = np.abs(stepList) < max_power*10**(-3) # limit ourselves to a 60dB range 
    stepList[threshold_mask] = max_power*10**(-3) 
    power = 20*np.log10(np.abs(stepList))
    return(power)

def log_spectral_power(stepList):
    yf = scipy.fft.fft(stepList, axis = 1)
    yf = np.abs(yf[:,:yf.shape[1]//2])
    max_power = np.max(np.abs(np.ravel(yf)))
    yf[yf < max_power*10**(-3)] = max_power*10**(-3) # limit ourselves to a 60dB range 
    power = 20*np.log10(np.abs(yf))
    return(power, yf)


def prob_calc(stepList, binNum = 10): # find probabilities of each frequency in each window
    max_val = np.max(np.ravel(stepList))
    probList = np.zeros((stepList.shape[0],binNum))
    for i in range(0, len(stepList)): # goes through each window
        m = stepList[i,:]
        hist, bin_edges = np.histogram(m, binNum, range=(max_val - 65, max_val)) 
        probList[i,:] = hist / sum(hist) # calculates the probability of each frequency bin
    return(probList)                                            
    
def entropy_calc(probList): # returns H(x) for every window
    hsum = np.zeros(len(probList))
    for i in range(0, len(probList)): # goes through each window
        hsum[i] +=  np.sum(-probList[i,:] * np.log2(probList[i,:], out=np.zeros_like(probList[i,:]), where=(probList[i,:] != 0))) # calculates Shannon entropy for each window
    return hsum

def kl_calc(probList, memory_decay_constant = 0):
    klist = np.zeros(len(probList) - 1)
    current_prob = probList[0,:]
    for i in range(0, len(probList) -1):
        kl = 0
        if i > 0:
            # define a weighting function
            ew = np.exp(memory_decay_constant*(np.arange(-i,0))) # this is 1 at the current time but decays exponentially into the past (unless memory_decay_constant = 0, in which case all past history is weighted equally)
            normalization = np.mean(ew)

            # print(ew)

            ew = np.outer(ew, np.ones(np.shape(probList)[1]))
            weighted_mean = np.multiply(probList[0:i,:], ew)
            # compute mean with weighting applied
            current_prob = np.mean(weighted_mean, axis = 0)/normalization 
        kl = np.sum(probList[i+1,:] * (
            np.log2(probList[i+1,:], out = np.zeros_like(current_prob), where=(probList[i+1,:] != 0))
            - np.log2(current_prob, out = np.zeros_like(current_prob), where=(current_prob != 0))  
            # np.log2(probList[i+1,:]) - np.log2(current_prob)  
        ))
        klist[i] = kl
    return(klist)     


### compute time weighted average ###
def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)

    alpha_rev = 1-alpha

    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]

    offset = data[0]*pows[1:]

    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr

    cumsums = mult.cumsum()

    out = offset + cumsums*scale_arr[::-1]

    return out

### graphing and visualization functions ###
def fft_plot(message, rate, size = (40, 10)):
    n = len(message)
    T = 1/ rate
    yf = scipy.fft.fft(message)
    xf = np.linspace(0., 1./(2.*T), int(n/2))
    fig, ax = plt.subplots(figsize = size)
    ax.plot(xf, 2. * np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Fast Fourier Transformation')
    plt.show()
    

def wave_plot(message, rate, size = (40, 10)):
    plt.figure(figsize = size)
    librosa.display.waveshow(y = message, sr = rate)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.show()


### Tim Sainburg Spectrogram Functions ###
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def overlap(X, window_size, window_step):
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = (valid) // ss
    out = np.ndarray((nw, ws), dtype=a.dtype)

    for i in np.arange(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start:stop]
    return out

def stft(
    X, fftsize=128, step=65, mean_normalize=True, real=False, compute_onesided=True
):
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()
    X = overlap(X, fftsize, step)
    size = fftsize
    win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X

