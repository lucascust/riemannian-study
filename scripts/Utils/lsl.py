import pylsl

import numpy as np

from pylsl import StreamInlet, resolve_stream
import keyboard
import time

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('name', 'openvibeSignal')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

sample, timestamp = inlet.pull_sample()
time_window = np.array(sample)
while not keyboard.is_pressed('s'):
    t1 = time.time()
    sample, timestamp = inlet.pull_sample()
    sample = np.array(sample)
    time_window = np.vstack((time_window, sample))
    t2 = time.time()
    print("time: ", t2-t1)

print(time_window)
