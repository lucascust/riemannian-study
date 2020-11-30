import pylsl

from pylsl import StreamInlet, resolve_stream

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('name', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
count=0
arr=[]
while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    print(timestamp, sample,count)
    arr.append(sample)
    count=count+1
print(len(arr))

print("Aqui são so dados")
print(arr)