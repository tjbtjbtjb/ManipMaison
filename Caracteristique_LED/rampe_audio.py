import sounddevice as sd
from scipy import signal
import numpy as np;
fs=44000;
t=np.arange(0,5,1/fs);
a=signal.sawtooth(t*2*np.pi*5);
sd.play(a,fs)
