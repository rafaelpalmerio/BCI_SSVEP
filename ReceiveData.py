import sys; sys.path.append('..')  # make sure that pylsl is found (note: in a normal program you would bundle pylsl with the program)
import pylsl
import time

# first resolve an EEG stream on the lab network
#print("looking for an EEG stream...")
#streams = pylsl.resolve_stream('type','EEG')

# create a new inlet to read from the stream
inlet = pylsl.stream_inlet(streams[0])

#sample = pylsl.vectorf()

#while True:
it = 0
vazio=0
#time.sleep(3)
now = time.time()

while time.time()-now <=1:
    # get a new sample (you can also omit the timestamp part if you're not interested in it)
    timestamp = inlet.pull_sample([])
    if timestamp[0]:
        it+=1
    else:
        vazio+=1
    #print(timestamp, list(sample))
    
print(it, vazio)