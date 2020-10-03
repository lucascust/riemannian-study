import pylsl
from pylsl import StreamInlet, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
from mne_realtime import lsl_client


# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
# stream name openvibeSignal
# TCP Port 15361
# port 1024

client = lsl_client.LSLClient(host = 'openvibeSignal', port = 1024, wait_max=30.0)
print(client.get_measurement_info())

client.start()
#client.start_receive_thread()
# create a new inlet to read from the stream
#inlet = StreamInlet(streams[0])